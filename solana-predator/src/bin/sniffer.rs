//! Solana Sniffer: Captures raw Raydium pool data via Helius Yellowstone gRPC.
//!
//! Usage:
//!   sniffer --endpoint <HELIUS_GRPC_URL> --pool <AMM_ADDRESS> \
//!           --coin-vault <VAULT_ADDRESS> --pc-vault <VAULT_ADDRESS> \
//!           --output market_vibrations.bin

use anyhow::{Context, Result};
use clap::Parser;
use futures::StreamExt;
use solana_predator::record::{write_record, Record};
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::BufWriter;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc;

use yellowstone_grpc_client::GeyserGrpcClient;
use yellowstone_grpc_proto::prelude::subscribe_update::UpdateOneof;
use yellowstone_grpc_proto::prelude::{
    CommitmentLevel, SubscribeRequest, SubscribeRequestFilterAccounts,
};

/// Offset of the `amount` field (u64 LE) in an SPL Token Account.
const TOKEN_AMOUNT_OFFSET: usize = 64;
/// Minimum SPL Token Account data size.
const TOKEN_ACCOUNT_SIZE: usize = 165;
/// Seconds to wait before reconnecting after a disconnect.
const RECONNECT_DELAY_SECS: u64 = 3;
/// Channel buffer size for the background writer.
const WRITER_CHANNEL_SIZE: usize = 4096;

#[derive(Parser)]
#[command(name = "sniffer", about = "Capture raw Raydium pool data")]
struct Args {
    /// Helius gRPC endpoint URL
    #[arg(long, env = "HELIUS_GRPC_URL")]
    endpoint: String,

    /// Helius API key (x-token for authentication)
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

/// Build the gRPC subscribe request for our three accounts.
fn build_request(args: &Args) -> SubscribeRequest {
    let mut accounts = HashMap::new();
    accounts.insert(
        "pool".to_string(),
        SubscribeRequestFilterAccounts {
            account: vec![
                args.pool.clone(),
                args.coin_vault.clone(),
                args.pc_vault.clone(),
            ],
            owner: vec![],
            filters: vec![],
            nonempty_txn_signature: None,
        },
    );

    SubscribeRequest {
        accounts,
        slots: HashMap::new(),
        transactions: HashMap::new(),
        transactions_status: HashMap::new(),
        blocks: HashMap::new(),
        blocks_meta: HashMap::new(),
        entry: HashMap::new(),
        commitment: Some(CommitmentLevel::Confirmed as i32),
        accounts_data_slice: vec![],
        ping: None,
        from_slot: None,
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    eprintln!("========================================");
    eprintln!("  SOLANA SNIFFER -- Data Capture");
    eprintln!("========================================");
    eprintln!("  Endpoint:   {}", args.endpoint);
    eprintln!("  Pool (AMM): {}", args.pool);
    eprintln!("  Coin vault: {}", args.coin_vault);
    eprintln!("  PC vault:   {}", args.pc_vault);
    eprintln!("  Output:     {}", args.output);
    eprintln!();

    // WARN if appending to an existing file (timeline warp risk between sessions).
    if std::path::Path::new(&args.output).exists() {
        let size = std::fs::metadata(&args.output).map(|m| m.len()).unwrap_or(0);
        if size > 0 {
            eprintln!("  WARNING: Output file already exists ({} bytes).", size);
            eprintln!("  New data will be APPENDED. If this is a separate session,");
            eprintln!("  the timeline gap may confuse training. Use a fresh filename");
            eprintln!("  or delete the old file first.");
            eprintln!();
        }
    }

    // Background writer thread: receives Records over a channel, writes to disk.
    // Network thread never touches the filesystem.
    let (tx, mut rx) = mpsc::channel::<Record>(WRITER_CHANNEL_SIZE);
    let output_path = args.output.clone();

    let _writer_handle = tokio::task::spawn_blocking(move || -> Result<()> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&output_path)
            .with_context(|| format!("Failed to open output file: {}", output_path))?;
        let mut writer = BufWriter::new(file);
        let mut count: u64 = 0;

        while let Some(record) = rx.blocking_recv() {
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

    // Track latest vault balances (updated asynchronously)
    let coin_amount = Arc::new(AtomicU64::new(0));
    let pc_amount = Arc::new(AtomicU64::new(0));
    let mut record_count: u64 = 0;
    let mut session: u64 = 0;

    // Reconnection loop: reconnect on any stream error or disconnect.
    loop {
        session += 1;
        eprintln!("[session {}] Connecting to gRPC endpoint...", session);

        let mut client = match GeyserGrpcClient::build_from_shared(args.endpoint.clone())
            .and_then(|b| b.x_token(args.api_key.clone()))
        {
            Ok(builder) => match builder.connect().await {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("[session {}] Connect failed: {}. Retrying in {}s...",
                        session, e, RECONNECT_DELAY_SECS);
                    tokio::time::sleep(std::time::Duration::from_secs(RECONNECT_DELAY_SECS)).await;
                    continue;
                }
            },
            Err(e) => {
                eprintln!("[session {}] Build failed: {}. Retrying in {}s...",
                    session, e, RECONNECT_DELAY_SECS);
                tokio::time::sleep(std::time::Duration::from_secs(RECONNECT_DELAY_SECS)).await;
                continue;
            }
        };
        eprintln!("[session {}] Connected!", session);

        let request = build_request(&args);
        let mut stream = match client.subscribe_once(request).await {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[session {}] Subscribe failed: {}. Retrying in {}s...",
                    session, e, RECONNECT_DELAY_SECS);
                tokio::time::sleep(std::time::Duration::from_secs(RECONNECT_DELAY_SECS)).await;
                continue;
            }
        };

        eprintln!("[session {}] Subscribed! Waiting for updates...", session);

        while let Some(message) = stream.next().await {
            let msg = match message {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("[session {}] Stream error: {}. Reconnecting...", session, e);
                    break;
                }
            };

            if let Some(update) = msg.update_oneof {
                if let UpdateOneof::Account(account_update) = update {
                    if let Some(account_info) = &account_update.account {
                        let pubkey_b58 = bs58::encode(&account_info.pubkey).into_string();
                        let slot = account_update.slot;
                        let data = &account_info.data;

                        if pubkey_b58 == args.coin_vault {
                            if data.len() >= TOKEN_ACCOUNT_SIZE {
                                if let Some(amount) = extract_token_amount(data) {
                                    coin_amount.store(amount, Ordering::Relaxed);
                                }
                            }
                        } else if pubkey_b58 == args.pc_vault {
                            if data.len() >= TOKEN_ACCOUNT_SIZE {
                                if let Some(amount) = extract_token_amount(data) {
                                    pc_amount.store(amount, Ordering::Relaxed);
                                }
                            }
                        } else if pubkey_b58 == args.pool {
                            let record = Record::from_raw(
                                slot,
                                data,
                                coin_amount.load(Ordering::Relaxed),
                                pc_amount.load(Ordering::Relaxed),
                            );

                            record_count += 1;
                            if record_count % 100 == 0 {
                                eprintln!(
                                    "  Records: {} | Slot: {} | Price: ~{:.2} USDC/SOL",
                                    record_count, slot, record.price(9, 6),
                                );
                            }

                            // Send to background writer — never blocks the network thread.
                            // If the channel is full, we drop the record and warn.
                            if tx.try_send(record).is_err() {
                                eprintln!("  WARNING: Writer backpressure, record dropped at slot {}", slot);
                            }
                        }
                    }
                }
            }
        }

        eprintln!("[session {}] Stream ended after {} records. Reconnecting in {}s...",
            session, record_count, RECONNECT_DELAY_SECS);
        tokio::time::sleep(std::time::Duration::from_secs(RECONNECT_DELAY_SECS)).await;
    }

    // Note: In practice we never reach here (Ctrl+C exits).
    // If we did, dropping tx closes the channel and the writer thread finishes.
    #[allow(unreachable_code)]
    {
        drop(tx);
        _writer_handle.await??;
        Ok(())
    }
}
