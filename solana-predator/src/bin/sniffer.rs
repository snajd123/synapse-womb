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
use std::fs::File;
use std::io::BufWriter;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use yellowstone_grpc_client::GeyserGrpcClient;
use yellowstone_grpc_proto::prelude::subscribe_update::UpdateOneof;
use yellowstone_grpc_proto::prelude::{
    CommitmentLevel, SubscribeRequest, SubscribeRequestFilterAccounts,
};

/// SPL Token Account data size.
const TOKEN_ACCOUNT_SIZE: usize = 165;
/// Offset of the `amount` field (u64 LE) in an SPL Token Account.
const TOKEN_AMOUNT_OFFSET: usize = 64;

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

    // Open output file
    let file = File::create(&args.output)
        .with_context(|| format!("Failed to create output file: {}", args.output))?;
    let mut writer = BufWriter::new(file);

    // Connect to Helius Yellowstone gRPC
    eprintln!("Connecting to gRPC endpoint...");
    let mut client = GeyserGrpcClient::build_from_shared(args.endpoint.clone())?
        .x_token(args.api_key.clone())?
        .connect()
        .await
        .with_context(|| "Failed to connect to gRPC endpoint")?;
    eprintln!("Connected!");

    // Subscribe to all three accounts: AMM + coin vault + PC vault
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

    let request = SubscribeRequest {
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
    };

    let mut stream = client
        .subscribe_once(request)
        .await
        .with_context(|| "Failed to subscribe")?;

    eprintln!("Subscribed! Waiting for updates... (Ctrl+C to stop)");

    // Track latest vault balances (updated asynchronously)
    let coin_amount = Arc::new(AtomicU64::new(0));
    let pc_amount = Arc::new(AtomicU64::new(0));
    let mut record_count: u64 = 0;

    while let Some(message) = stream.next().await {
        let msg = message.with_context(|| "Stream error")?;

        if let Some(update) = msg.update_oneof {
            if let UpdateOneof::Account(account_update) = update {
                if let Some(account_info) = &account_update.account {
                    let pubkey_b58 = bs58::encode(&account_info.pubkey).into_string();
                    let slot = account_update.slot;
                    let data = &account_info.data;

                    if pubkey_b58 == args.coin_vault {
                        // Update coin vault balance
                        if data.len() >= TOKEN_ACCOUNT_SIZE {
                            if let Some(amount) = extract_token_amount(data) {
                                coin_amount.store(amount, Ordering::Relaxed);
                            }
                        }
                    } else if pubkey_b58 == args.pc_vault {
                        // Update PC vault balance
                        if data.len() >= TOKEN_ACCOUNT_SIZE {
                            if let Some(amount) = extract_token_amount(data) {
                                pc_amount.store(amount, Ordering::Relaxed);
                            }
                        }
                    } else if pubkey_b58 == args.pool {
                        // AMM account update -- write a record
                        let record = Record::from_raw(
                            slot,
                            data,
                            coin_amount.load(Ordering::Relaxed),
                            pc_amount.load(Ordering::Relaxed),
                        );

                        write_record(&mut writer, &record)
                            .with_context(|| "Failed to write record")?;

                        record_count += 1;
                        if record_count % 100 == 0 {
                            eprintln!(
                                "  Records: {} | Slot: {} | Price: ~{:.2} USDC/SOL",
                                record_count,
                                slot,
                                record.price(9, 6),
                            );
                        }
                    }
                }
            }
        }
    }

    eprintln!("\nStream ended. Total records: {}", record_count);
    Ok(())
}
