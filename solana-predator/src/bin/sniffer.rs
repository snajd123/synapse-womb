//! Solana Sniffer: Captures raw Raydium pool data via WebSocket or gRPC.
//!
//! Usage:
//!   sniffer --source ws --endpoint <WS_URL> --pool <AMM_ADDRESS> \
//!           --coin-vault <VAULT_ADDRESS> --pc-vault <VAULT_ADDRESS> \
//!           --output market_vibrations.bin

use anyhow::{bail, Context, Result};
use clap::Parser;
use solana_predator::record::{write_record, Record};
use solana_predator::source::{AccountUpdate, MarketConfig, Source};
use std::fs::OpenOptions;
use std::io::BufWriter;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc;

/// Offset of the `amount` field (u64 LE) in an SPL Token Account.
const TOKEN_AMOUNT_OFFSET: usize = 64;
/// Minimum SPL Token Account data size.
const TOKEN_ACCOUNT_SIZE: usize = 165;
/// Seconds to wait before reconnecting after a disconnect.
const RECONNECT_DELAY_SECS: u64 = 3;
/// Channel buffer size for account updates from the source.
const UPDATE_CHANNEL_SIZE: usize = 4096;
/// Channel buffer size for the background writer.
const WRITER_CHANNEL_SIZE: usize = 4096;

#[derive(Parser)]
#[command(name = "sniffer", about = "Capture raw Raydium pool data")]
struct Args {
    /// Data source: "ws" (WebSocket, default) or "grpc" (Yellowstone)
    #[arg(long, default_value = "ws")]
    source: String,

    /// Endpoint URL (wss:// for WebSocket, https:// for gRPC)
    #[arg(long, env = "HELIUS_ENDPOINT")]
    endpoint: String,

    /// API key (x-token for gRPC; for WebSocket, bake into URL query param)
    #[arg(long, env = "HELIUS_API_KEY")]
    api_key: Option<String>,

    /// Raydium AMM account address (base58)
    #[arg(long, default_value = "58oQChX4yWmvJFWGf6JuSCniYcxPdGWoTcue7CKRfyuY")]
    pool: String,

    /// SOL vault token account address (base58)
    #[arg(long)]
    coin_vault: String,

    /// USDC vault token account address (base58)
    #[arg(long)]
    pc_vault: String,

    /// Output file path
    #[arg(long, short, default_value = "market_vibrations.bin")]
    output: String,
}

/// Extract u64 amount from SPL Token account data at the standard offset.
fn extract_token_amount(data: &[u8]) -> Option<u64> {
    if data.len() < TOKEN_AMOUNT_OFFSET + 8 {
        return None;
    }
    Some(u64::from_le_bytes(
        data[TOKEN_AMOUNT_OFFSET..TOKEN_AMOUNT_OFFSET + 8]
            .try_into()
            .ok()?,
    ))
}

fn build_source(name: &str) -> Result<Source> {
    match name {
        #[cfg(feature = "websocket")]
        "ws" | "websocket" => {
            Ok(Source::WebSocket(solana_predator::source::WebSocketSource::new()))
        }
        #[cfg(feature = "grpc")]
        "grpc" => {
            Ok(Source::Grpc(solana_predator::source::GrpcSource::new()))
        }
        other => {
            let mut supported = Vec::new();
            #[cfg(feature = "websocket")]
            supported.push("ws");
            #[cfg(feature = "grpc")]
            supported.push("grpc");
            bail!(
                "Unknown source {:?}. Supported (compiled): {:?}",
                other,
                supported
            );
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let source = build_source(&args.source)?;

    eprintln!("========================================");
    eprintln!("  SOLANA SNIFFER -- Data Capture");
    eprintln!("========================================");
    eprintln!("  Source:     {}", args.source);
    eprintln!("  Endpoint:   {}", args.endpoint);
    eprintln!("  Pool (AMM): {}", args.pool);
    eprintln!("  Coin vault: {}", args.coin_vault);
    eprintln!("  PC vault:   {}", args.pc_vault);
    eprintln!("  Output:     {}", args.output);
    eprintln!();

    // WARN if appending to an existing file (timeline warp risk between sessions).
    if std::path::Path::new(&args.output).exists() {
        let size = std::fs::metadata(&args.output)
            .map(|m| m.len())
            .unwrap_or(0);
        if size > 0 {
            eprintln!("  WARNING: Output file already exists ({} bytes).", size);
            eprintln!("  New data will be APPENDED. If this is a separate session,");
            eprintln!("  the timeline gap may confuse training. Use a fresh filename");
            eprintln!("  or delete the old file first.");
            eprintln!();
        }
    }

    let config = MarketConfig {
        endpoint: args.endpoint.clone(),
        api_key: args.api_key.clone(),
        accounts: vec![
            args.pool.clone(),
            args.coin_vault.clone(),
            args.pc_vault.clone(),
        ],
    };

    // Background writer thread: receives Records over a channel, writes to disk.
    let (record_tx, mut record_rx) = mpsc::channel::<Record>(WRITER_CHANNEL_SIZE);
    let output_path = args.output.clone();

    let _writer_handle = tokio::task::spawn_blocking(move || -> Result<()> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&output_path)
            .with_context(|| format!("Failed to open output file: {}", output_path))?;
        let mut writer = BufWriter::new(file);
        let mut count: u64 = 0;

        while let Some(record) = record_rx.blocking_recv() {
            write_record(&mut writer, &record)
                .with_context(|| "Failed to write record")?;
            count += 1;
            if count % 100 == 0 {
                eprintln!("  [writer] {} records flushed to disk", count);
            }
        }

        eprintln!("  [writer] Channel closed. Total written: {}", count);
        Ok(())
    });

    // Track latest vault balances (updated asynchronously).
    let coin_amount = Arc::new(AtomicU64::new(0));
    let pc_amount = Arc::new(AtomicU64::new(0));
    let mut record_count: u64 = 0;
    let mut session: u64 = 0;

    // Reconnection loop: reconnect on any stream error or disconnect.
    loop {
        session += 1;
        eprintln!("[session {}] Connecting via {}...", session, args.source);

        let (update_tx, mut update_rx) = mpsc::channel::<AccountUpdate>(UPDATE_CHANNEL_SIZE);

        let source_clone = source.clone();
        let config_ref = MarketConfig {
            endpoint: config.endpoint.clone(),
            api_key: config.api_key.clone(),
            accounts: config.accounts.clone(),
        };

        // Spawn the source stream in a task.
        let stream_handle = tokio::spawn(async move {
            source_clone.stream(&config_ref, update_tx).await
        });

        eprintln!("[session {}] Subscribed! Waiting for updates...", session);

        // Process updates from the source (source-agnostic).
        while let Some(update) = update_rx.recv().await {
            if update.pubkey == args.coin_vault {
                if update.data.len() >= TOKEN_ACCOUNT_SIZE {
                    if let Some(amount) = extract_token_amount(&update.data) {
                        coin_amount.store(amount, Ordering::Relaxed);
                    }
                }
            } else if update.pubkey == args.pc_vault {
                if update.data.len() >= TOKEN_ACCOUNT_SIZE {
                    if let Some(amount) = extract_token_amount(&update.data) {
                        pc_amount.store(amount, Ordering::Relaxed);
                    }
                }
            } else if update.pubkey == args.pool {
                let record = Record::from_raw(
                    update.slot,
                    &update.data,
                    coin_amount.load(Ordering::Relaxed),
                    pc_amount.load(Ordering::Relaxed),
                );

                record_count += 1;
                if record_count % 100 == 0 {
                    eprintln!(
                        "  Records: {} | Slot: {} | Price: ~{:.2} USDC/SOL",
                        record_count,
                        update.slot,
                        record.price(9, 6),
                    );
                }

                if record_tx.try_send(record).is_err() {
                    eprintln!(
                        "  WARNING: Writer backpressure, record dropped at slot {}",
                        update.slot
                    );
                }
            }
        }

        // Channel closed — source disconnected or errored.
        if let Ok(Err(e)) = stream_handle.await {
            eprintln!("[session {}] Source error: {}.", session, e);
        }

        eprintln!(
            "[session {}] Stream ended after {} records. Reconnecting in {}s...",
            session, record_count, RECONNECT_DELAY_SECS
        );
        tokio::time::sleep(std::time::Duration::from_secs(RECONNECT_DELAY_SECS)).await;
    }

    #[allow(unreachable_code)]
    {
        drop(record_tx);
        _writer_handle.await??;
        Ok(())
    }
}
