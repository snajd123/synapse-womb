//! One-off tool: fetch Orca Whirlpool SOL/USDC pool account and extract vault pubkeys.
//! Pure Rust — no Python, no SDK. Just HTTP + base64 + bs58.

use anyhow::{bail, Result};
use base64::Engine;

const ORCA_POOL: &str = "HJPjoWUrhoZzkNfRpHuieeFk9WcZWjwy6PBjZ81ngndJ";
const RPC_URL: &str = "https://api.mainnet-beta.solana.com";
// Whirlpool layout (after 8-byte Anchor discriminator):
// token_vault_a: Pubkey (32 bytes) at struct offset 125 → raw offset 133
// token_vault_b: Pubkey (32 bytes) at struct offset 205 → raw offset 213
const VAULT_A_OFFSET: usize = 133;
const VAULT_B_OFFSET: usize = 213;

#[tokio::main]
async fn main() -> Result<()> {
    eprintln!("Fetching Orca Whirlpool: {}", ORCA_POOL);

    let client = reqwest::Client::new();
    let body = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getAccountInfo",
        "params": [ORCA_POOL, {"encoding": "base64"}]
    });

    let resp: serde_json::Value = client.post(RPC_URL)
        .json(&body)
        .send()
        .await?
        .json()
        .await?;

    let data_b64 = resp["result"]["value"]["data"][0]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("No account data"))?;

    let raw = base64::engine::general_purpose::STANDARD.decode(data_b64)?;

    eprintln!("Account size: {} bytes", raw.len());
    if raw.len() < VAULT_B_OFFSET + 32 {
        bail!("Account too small: {} < {}", raw.len(), VAULT_B_OFFSET + 32);
    }

    let vault_a = bs58::encode(&raw[VAULT_A_OFFSET..VAULT_A_OFFSET + 32]).into_string();
    let vault_b = bs58::encode(&raw[VAULT_B_OFFSET..VAULT_B_OFFSET + 32]).into_string();

    eprintln!("token_vault_a (SOL):  {}", vault_a);
    eprintln!("token_vault_b (USDC): {}", vault_b);

    // Verify each vault is an SPL Token account
    for (label, addr) in [("vault_a", &vault_a), ("vault_b", &vault_b)] {
        let body = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getAccountInfo",
            "params": [addr, {"encoding": "jsonParsed"}]
        });
        let resp: serde_json::Value = client.post(RPC_URL)
            .json(&body)
            .send()
            .await?
            .json()
            .await?;

        let owner = resp["result"]["value"]["owner"].as_str().unwrap_or("???");
        let program = resp["result"]["value"]["data"]["program"].as_str().unwrap_or("???");
        let amount = resp["result"]["value"]["data"]["parsed"]["info"]["tokenAmount"]["uiAmountString"]
            .as_str().unwrap_or("???");
        eprintln!("  {}: owner={} program={} balance={}", label, owner, program, amount);
    }

    // Print final addresses for copy-paste
    println!("ORCA_COIN_VAULT={}", vault_a);
    println!("ORCA_PC_VAULT={}", vault_b);

    Ok(())
}
