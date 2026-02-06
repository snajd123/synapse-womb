//! Solana Sniffer: Captures raw Raydium pool data via WebSocket or gRPC.
//!
//! Usage:
//!   sniffer --source ws --endpoint <WS_URL> --pool <AMM_ADDRESS> \
//!           --coin-vault <VAULT_ADDRESS> --pc-vault <VAULT_ADDRESS> \
//!           --output market_vibrations.bin

use anyhow::{bail, Context, Result};
use clap::Parser;
use solana_predator::record::{write_record, write_dual_record, Record, DualRecord, AMM_DATA_SIZE};
use solana_predator::source::{AccountUpdate, MarketConfig, Source};
use std::fs::OpenOptions;
use std::io::BufWriter;
use tokio::sync::mpsc;

/// Offset of the `amount` field (u64 LE) in an SPL Token Account.
const TOKEN_AMOUNT_OFFSET: usize = 64;
/// Minimum SPL Token Account data size (Legacy=165, Token-2022 can be 82).
const TOKEN_ACCOUNT_SIZE: usize = 82;
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

    /// Orca Whirlpool account address (base58). If set, enables dual-pool mode.
    #[arg(long)]
    orca_pool: Option<String>,

    /// Orca SOL vault token account (base58). Required if --orca-pool is set.
    #[arg(long)]
    orca_coin_vault: Option<String>,

    /// Orca USDC vault token account (base58). Required if --orca-pool is set.
    #[arg(long)]
    orca_pc_vault: Option<String>,

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

/// Emit a Record from the buffered AMM update and current vault balances.
fn flush_record(
    pending_amm: &mut Option<(u64, Vec<u8>)>,
    coin_balance: u64,
    pc_balance: u64,
    record_tx: &mpsc::Sender<RecordMessage>,
    record_count: &mut u64,
) {
    if let Some((slot, data)) = pending_amm.take() {
        let record = Record::from_raw(slot, &data, coin_balance, pc_balance);

        *record_count += 1;
        if *record_count % 100 == 0 {
            eprintln!(
                "  Records: {} | Slot: {} | Price: ~{:.2} USDC/SOL",
                record_count, slot, record.price(9, 6),
            );
        }

        if record_tx.try_send(RecordMessage::Single(record)).is_err() {
            eprintln!(
                "  WARNING: Writer backpressure, record dropped at slot {}",
                slot
            );
        }
    }
}

/// Message sent to the background writer (either single or dual record).
enum RecordMessage {
    Single(Record),
    Dual(DualRecord),
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

    let dual_mode = args.orca_pool.is_some();
    if dual_mode {
        if args.orca_coin_vault.is_none() || args.orca_pc_vault.is_none() {
            bail!("--orca-pool requires --orca-coin-vault and --orca-pc-vault");
        }
    }

    eprintln!("========================================");
    eprintln!("  SOLANA SNIFFER -- Data Capture");
    eprintln!("========================================");
    eprintln!("  Source:     {}", args.source);
    eprintln!("  Endpoint:   {}", args.endpoint);
    eprintln!("  Mode:       {}", if dual_mode { "DUAL (Raydium + Orca)" } else { "SINGLE (Raydium)" });
    eprintln!("  Pool (AMM): {}", args.pool);
    eprintln!("  Coin vault: {}", args.coin_vault);
    eprintln!("  PC vault:   {}", args.pc_vault);
    if dual_mode {
        eprintln!("  Orca pool:  {}", args.orca_pool.as_ref().unwrap());
        eprintln!("  Orca coin:  {}", args.orca_coin_vault.as_ref().unwrap());
        eprintln!("  Orca pc:    {}", args.orca_pc_vault.as_ref().unwrap());
    }
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

    let mut accounts = vec![
        args.pool.clone(),
        args.coin_vault.clone(),
        args.pc_vault.clone(),
    ];
    if dual_mode {
        accounts.push(args.orca_pool.as_ref().unwrap().clone());
        accounts.push(args.orca_coin_vault.as_ref().unwrap().clone());
        accounts.push(args.orca_pc_vault.as_ref().unwrap().clone());
    }
    let config = MarketConfig {
        endpoint: args.endpoint.clone(),
        api_key: args.api_key.clone(),
        accounts,
    };

    // Background writer thread: receives Records over a channel, writes to disk.
    let (record_tx, mut record_rx) = mpsc::channel::<RecordMessage>(WRITER_CHANNEL_SIZE);
    let output_path = args.output.clone();

    let _writer_handle = tokio::task::spawn_blocking(move || -> Result<()> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&output_path)
            .with_context(|| format!("Failed to open output file: {}", output_path))?;
        let mut writer = BufWriter::new(file);
        let mut count: u64 = 0;

        while let Some(msg) = record_rx.blocking_recv() {
            match msg {
                RecordMessage::Single(record) => {
                    write_record(&mut writer, &record)
                        .with_context(|| "Failed to write record")?;
                }
                RecordMessage::Dual(dual) => {
                    write_dual_record(&mut writer, &dual)
                        .with_context(|| "Failed to write dual record")?;
                }
            }
            count += 1;
            if count % 100 == 0 {
                eprintln!("  [writer] {} records flushed to disk", count);
            }
        }

        eprintln!("  [writer] Channel closed. Total written: {}", count);
        Ok(())
    });

    // Vault balances persist across reconnects (still valid if unchanged).
    let mut coin_balance: u64 = 0;
    let mut pc_balance: u64 = 0;
    let mut record_count: u64 = 0;
    let mut session: u64 = 0;

    // Reconnection loop: reconnect on any stream error or disconnect.
    loop {
        session += 1;
        eprintln!("[session {}] Connecting via {}...", session, args.source);

        // Slot tracking and pending AMM reset per session — slot numbers
        // from the old stream are meaningless after reconnect.
        let mut coin_slot: u64 = 0;
        let mut pc_slot: u64 = 0;
        let mut pending_amm: Option<(u64, Vec<u8>)> = None;

        // Orca state (only used in dual mode)
        let mut orca_pending: Option<(u64, Vec<u8>)> = None;
        let mut orca_coin_balance: u64 = 0;
        let mut orca_pc_balance: u64 = 0;
        let mut orca_coin_slot: u64 = 0;
        let mut orca_pc_slot: u64 = 0;

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
            // Route update to the correct handler via a single if/else chain
            // to avoid moving `update.data` twice.
            if update.pubkey == args.coin_vault {
                if update.data.len() >= TOKEN_ACCOUNT_SIZE {
                    if let Some(amount) = extract_token_amount(&update.data) {
                        coin_balance = amount;
                        coin_slot = update.slot;
                    }
                }
            } else if update.pubkey == args.pc_vault {
                if update.data.len() >= TOKEN_ACCOUNT_SIZE {
                    if let Some(amount) = extract_token_amount(&update.data) {
                        pc_balance = amount;
                        pc_slot = update.slot;
                    }
                }
            } else if update.pubkey == args.pool {
                if !dual_mode && pending_amm.is_some() {
                    flush_record(
                        &mut pending_amm,
                        coin_balance,
                        pc_balance,
                        &record_tx,
                        &mut record_count,
                    );
                }
                pending_amm = Some((update.slot, update.data));
            } else if dual_mode {
                // Orca updates (only reachable in dual mode)
                let orca_cv = args.orca_coin_vault.as_ref().unwrap();
                let orca_pv = args.orca_pc_vault.as_ref().unwrap();
                let orca_pool_addr = args.orca_pool.as_ref().unwrap();

                if update.pubkey == *orca_cv {
                    if update.data.len() >= TOKEN_ACCOUNT_SIZE {
                        if let Some(amount) = extract_token_amount(&update.data) {
                            orca_coin_balance = amount;
                            orca_coin_slot = update.slot;
                        }
                    }
                } else if update.pubkey == *orca_pv {
                    if update.data.len() >= TOKEN_ACCOUNT_SIZE {
                        if let Some(amount) = extract_token_amount(&update.data) {
                            orca_pc_balance = amount;
                            orca_pc_slot = update.slot;
                        }
                    }
                } else if update.pubkey == *orca_pool_addr {
                    orca_pending = Some((update.slot, update.data));
                }
            }

            // --- Flush logic ---
            if dual_mode {
                // Dual-pool flush: both pools must have pending + vaults caught up
                if let (Some((ray_slot, _)), Some((orca_slot, _))) = (&pending_amm, &orca_pending) {
                    let all_caught_up = coin_slot >= *ray_slot
                        && pc_slot >= *ray_slot
                        && orca_coin_slot >= *orca_slot
                        && orca_pc_slot >= *orca_slot;

                    if all_caught_up {
                        let (ray_s, ray_data) = pending_amm.take().unwrap();
                        let (_orca_s, orca_data) = orca_pending.take().unwrap();
                        let dr = DualRecord {
                            slot: ray_s,
                            ray_amm_data: {
                                let mut d = [0u8; AMM_DATA_SIZE];
                                let len = ray_data.len().min(AMM_DATA_SIZE);
                                d[..len].copy_from_slice(&ray_data[..len]);
                                d
                            },
                            ray_coin: coin_balance,
                            ray_pc: pc_balance,
                            orca_data: {
                                let mut d = [0u8; AMM_DATA_SIZE];
                                let len = orca_data.len().min(AMM_DATA_SIZE);
                                d[..len].copy_from_slice(&orca_data[..len]);
                                d
                            },
                            orca_coin: orca_coin_balance,
                            orca_pc: orca_pc_balance,
                        };

                        record_count += 1;
                        if record_count % 100 == 0 {
                            eprintln!(
                                "  DualRecords: {} | Slot: {} | Ray: ~{:.2} | Orca: ~{:.2}",
                                record_count, ray_s,
                                dr.ray_price(9, 6), dr.orca_price(9, 6),
                            );
                        }

                        if record_tx.try_send(RecordMessage::Dual(dr)).is_err() {
                            eprintln!(
                                "  WARNING: Writer backpressure, dual record dropped at slot {}",
                                ray_s
                            );
                        }
                    }
                }
            } else {
                // Single-pool flush: flush pending AMM if vaults have caught up
                if let Some((amm_slot, _)) = &pending_amm {
                    if coin_slot >= *amm_slot && pc_slot >= *amm_slot {
                        flush_record(
                            &mut pending_amm,
                            coin_balance,
                            pc_balance,
                            &record_tx,
                            &mut record_count,
                        );
                    }
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
