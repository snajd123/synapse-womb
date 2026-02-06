//! Fetch historical account data by polling Solana JSON-RPC.
//!
//! Calls `getMultipleAccounts` to snapshot the AMM + vault accounts,
//! then writes Records in the exact same binary format as the sniffer.
//!
//! Usage:
//!   fetch_history --endpoint https://mainnet.helius-rpc.com/?api-key=XXX \
//!                 --pool 58oQChX4yWmvJFWGf6JuSCniYcxPdGWoTcue7CKRfyuY \
//!                 --coin-vault <ADDR> --pc-vault <ADDR> \
//!                 --output market_vibrations.bin --records 1000

use anyhow::{anyhow, bail, Context, Result};
use base64::Engine;
use clap::Parser;
use serde::{Deserialize, Serialize};
use solana_predator::record::{write_record, Record};
use std::fs::OpenOptions;
use std::io::BufWriter;
use std::time::Duration;

/// Offset of the `amount` field (u64 LE) in an SPL Token Account.
const TOKEN_AMOUNT_OFFSET: usize = 64;
/// Minimum SPL Token Account data size.
const TOKEN_ACCOUNT_SIZE: usize = 165;

#[derive(Parser)]
#[command(name = "fetch_history", about = "Poll Solana RPC for account snapshots")]
struct Args {
    /// Solana JSON-RPC endpoint (HTTP/HTTPS)
    #[arg(long, env = "HELIUS_ENDPOINT")]
    endpoint: String,

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

    /// Number of unique records to capture
    #[arg(long, short, default_value = "1000")]
    records: usize,

    /// Polling interval in milliseconds
    #[arg(long, default_value = "400")]
    interval: u64,
}

// -- JSON-RPC plumbing --

#[derive(Serialize)]
struct RpcRequest {
    jsonrpc: &'static str,
    id: u64,
    method: String,
    params: serde_json::Value,
}

#[derive(Deserialize)]
struct RpcResponse {
    result: Option<serde_json::Value>,
    error: Option<RpcError>,
}

#[derive(Deserialize)]
struct RpcError {
    code: i64,
    message: String,
}

async fn rpc_call(
    client: &reqwest::Client,
    endpoint: &str,
    method: &str,
    params: serde_json::Value,
) -> Result<serde_json::Value> {
    let req = RpcRequest {
        jsonrpc: "2.0",
        id: 1,
        method: method.to_string(),
        params,
    };
    let resp: RpcResponse = client
        .post(endpoint)
        .json(&req)
        .send()
        .await
        .context("HTTP request failed")?
        .json()
        .await
        .context("Failed to parse RPC response")?;

    if let Some(err) = resp.error {
        bail!("RPC error {}: {}", err.code, err.message);
    }
    resp.result.ok_or_else(|| anyhow!("RPC returned null result"))
}

/// Decode base64 account data from a getMultipleAccounts response entry.
fn decode_account_data(entry: &serde_json::Value) -> Result<Vec<u8>> {
    let data_arr = entry["data"]
        .as_array()
        .ok_or_else(|| anyhow!("missing data field"))?;
    let b64 = data_arr[0]
        .as_str()
        .ok_or_else(|| anyhow!("data[0] not a string"))?;
    base64::engine::general_purpose::STANDARD
        .decode(b64)
        .context("base64 decode failed")
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
    let client = reqwest::Client::new();

    eprintln!("========================================");
    eprintln!("  FETCH HISTORY -- Account Snapshots");
    eprintln!("========================================");
    eprintln!("  Endpoint:   {}", args.endpoint);
    eprintln!("  Pool (AMM): {}", args.pool);
    eprintln!("  Coin vault: {}", args.coin_vault);
    eprintln!("  PC vault:   {}", args.pc_vault);
    eprintln!("  Output:     {}", args.output);
    eprintln!("  Records:    {}", args.records);
    eprintln!("  Interval:   {}ms", args.interval);
    eprintln!();

    // Verify endpoint works with a getSlot call.
    let slot_result = rpc_call(&client, &args.endpoint, "getSlot", serde_json::json!([]))
        .await
        .context("Failed initial getSlot — check your endpoint URL")?;
    eprintln!("  Connected. Current slot: {}", slot_result);
    eprintln!();

    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&args.output)
        .with_context(|| format!("Failed to open output: {}", args.output))?;
    let mut writer = BufWriter::new(file);

    let mut last_slot: u64 = 0;
    let mut written: usize = 0;
    let mut polls: usize = 0;
    let mut consecutive_skips: usize = 0;
    let start = std::time::Instant::now();

    while written < args.records {
        polls += 1;

        let result = match rpc_call(
            &client,
            &args.endpoint,
            "getMultipleAccounts",
            serde_json::json!([
                [&args.pool, &args.coin_vault, &args.pc_vault],
                {"encoding": "base64", "commitment": "confirmed"}
            ]),
        )
        .await
        {
            Ok(r) => r,
            Err(e) => {
                eprintln!("  [poll {}] RPC error: {}. Retrying...", polls, e);
                tokio::time::sleep(Duration::from_millis(args.interval * 2)).await;
                continue;
            }
        };

        let slot = result["context"]["slot"].as_u64().unwrap_or(0);

        // Skip duplicate slots — no state change.
        if slot == last_slot {
            consecutive_skips += 1;
            if consecutive_skips == 50 {
                eprintln!(
                    "  [poll {}] 50 consecutive skips — slot {} unchanged. Pool may be inactive.",
                    polls, slot
                );
            }
            tokio::time::sleep(Duration::from_millis(args.interval)).await;
            continue;
        }
        last_slot = slot;
        consecutive_skips = 0;

        let accounts = match result["value"].as_array() {
            Some(a) if a.len() == 3 => a,
            _ => {
                eprintln!("  [poll {}] Unexpected response shape, skipping", polls);
                tokio::time::sleep(Duration::from_millis(args.interval)).await;
                continue;
            }
        };

        // Check for null accounts (address doesn't exist on-chain).
        if accounts.iter().any(|a| a.is_null()) {
            let which: Vec<&str> = ["pool", "coin_vault", "pc_vault"]
                .iter()
                .zip(accounts.iter())
                .filter(|(_, a)| a.is_null())
                .map(|(name, _)| *name)
                .collect();
            bail!(
                "Account(s) not found on-chain: {:?}. Check your addresses.",
                which
            );
        }

        let amm_data = decode_account_data(&accounts[0])
            .context("Failed to decode AMM data")?;
        let coin_data = decode_account_data(&accounts[1])
            .context("Failed to decode coin vault data")?;
        let pc_data = decode_account_data(&accounts[2])
            .context("Failed to decode pc vault data")?;

        if coin_data.len() < TOKEN_ACCOUNT_SIZE {
            bail!(
                "Coin vault data too small ({} bytes). Expected >= {}. Wrong address?",
                coin_data.len(),
                TOKEN_ACCOUNT_SIZE
            );
        }
        if pc_data.len() < TOKEN_ACCOUNT_SIZE {
            bail!(
                "PC vault data too small ({} bytes). Expected >= {}. Wrong address?",
                pc_data.len(),
                TOKEN_ACCOUNT_SIZE
            );
        }

        let coin_amount = extract_token_amount(&coin_data).unwrap_or(0);
        let pc_amount = extract_token_amount(&pc_data).unwrap_or(0);

        // Record::from_raw handles truncation/zero-padding to 1024 bytes.
        let record = Record::from_raw(slot, &amm_data, coin_amount, pc_amount);
        write_record(&mut writer, &record)
            .context("Failed to write record")?;

        written += 1;

        if written % 10 == 0 || written == 1 {
            let elapsed = start.elapsed().as_secs();
            eprintln!(
                "  [{}/{}] Slot {} | Price: ~{:.4} USDC/SOL | {}s elapsed",
                written,
                args.records,
                slot,
                record.price(9, 6),
                elapsed,
            );
        }

        if written < args.records {
            tokio::time::sleep(Duration::from_millis(args.interval)).await;
        }
    }

    let elapsed = start.elapsed();
    eprintln!();
    eprintln!("========================================");
    eprintln!("  Done.");
    eprintln!("  Records written: {}", written);
    eprintln!("  Total polls:     {}", polls);
    eprintln!("  Time:            {:.1}s", elapsed.as_secs_f64());
    eprintln!("  Output:          {}", args.output);
    eprintln!("========================================");

    Ok(())
}
