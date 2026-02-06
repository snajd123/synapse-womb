//! WebSocket source using Solana RPC PubSub `accountSubscribe`.
//!
//! Uses raw `tokio-tungstenite` with manual JSON-RPC 2.0 to avoid
//! pulling in the full Solana SDK.

use anyhow::{anyhow, Context, Result};
use base64::Engine;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_tungstenite::{connect_async, tungstenite::Message};

use super::{AccountUpdate, MarketConfig};

/// JSON-RPC 2.0 request for `accountSubscribe`.
#[derive(Serialize)]
struct SubscribeRequest {
    jsonrpc: &'static str,
    id: u64,
    method: &'static str,
    params: (String, SubscribeParams),
}

#[derive(Serialize)]
struct SubscribeParams {
    encoding: &'static str,
    commitment: &'static str,
}

/// JSON-RPC 2.0 response to a subscribe request.
#[derive(Deserialize)]
struct SubscribeResponse {
    id: u64,
    result: Option<u64>,
    error: Option<RpcError>,
}

#[derive(Deserialize)]
struct RpcError {
    message: String,
}

/// JSON-RPC 2.0 notification for account updates.
#[derive(Deserialize)]
struct Notification {
    method: Option<String>,
    params: Option<NotificationParams>,
}

#[derive(Deserialize)]
struct NotificationParams {
    subscription: u64,
    result: NotificationResult,
}

#[derive(Deserialize)]
struct NotificationResult {
    context: SlotContext,
    value: AccountValue,
}

#[derive(Deserialize)]
struct SlotContext {
    slot: u64,
}

#[derive(Deserialize)]
struct AccountValue {
    data: (String, String), // (base64_data, encoding)
}

#[derive(Clone)]
pub struct WebSocketSource;

impl WebSocketSource {
    pub fn new() -> Self {
        Self
    }

    /// Connect to the WebSocket endpoint, subscribe to all accounts in `config`,
    /// and push `AccountUpdate`s into `tx` until the connection drops.
    pub async fn stream(
        &self,
        config: &MarketConfig,
        tx: mpsc::Sender<AccountUpdate>,
    ) -> Result<()> {
        let (ws, _resp) = connect_async(&config.endpoint)
            .await
            .context("WebSocket connect failed")?;

        let (mut write, mut read) = ws.split();

        // Send accountSubscribe for each account.
        // Request IDs are 1-indexed, matching the order in config.accounts.
        use futures::SinkExt;
        for (i, account) in config.accounts.iter().enumerate() {
            let req = SubscribeRequest {
                jsonrpc: "2.0",
                id: (i + 1) as u64,
                method: "accountSubscribe",
                params: (
                    account.clone(),
                    SubscribeParams {
                        encoding: "base64",
                        commitment: "confirmed",
                    },
                ),
            };
            let msg = serde_json::to_string(&req)?;
            write.send(Message::Text(msg)).await?;
        }

        // Collect subscription ID → pubkey mapping from responses.
        let mut sub_to_pubkey: std::collections::HashMap<u64, String> =
            std::collections::HashMap::new();
        let mut responses_needed = config.accounts.len();

        while let Some(msg) = read.next().await {
            let msg = msg.context("WebSocket read error")?;
            let text = match msg {
                Message::Text(t) => t,
                Message::Ping(_) | Message::Pong(_) => continue,
                Message::Close(_) => return Ok(()),
                _ => continue,
            };

            // Try parsing as a subscribe response first.
            if responses_needed > 0 {
                if let Ok(resp) = serde_json::from_str::<SubscribeResponse>(&text) {
                    if let Some(err) = resp.error {
                        return Err(anyhow!(
                            "accountSubscribe failed for request {}: {}",
                            resp.id,
                            err.message
                        ));
                    }
                    if let Some(sub_id) = resp.result {
                        let idx = (resp.id - 1) as usize;
                        if let Some(pubkey) = config.accounts.get(idx) {
                            sub_to_pubkey.insert(sub_id, pubkey.clone());
                        }
                    }
                    responses_needed -= 1;
                    continue;
                }
            }

            // Parse as notification.
            let notif: Notification = match serde_json::from_str(&text) {
                Ok(n) => n,
                Err(_) => continue,
            };

            if notif.method.as_deref() != Some("accountNotification") {
                continue;
            }

            let params = match notif.params {
                Some(p) => p,
                None => continue,
            };

            let pubkey = match sub_to_pubkey.get(&params.subscription) {
                Some(pk) => pk.clone(),
                None => continue,
            };

            let (b64_data, _encoding) = &params.result.value.data;
            let data = base64::engine::general_purpose::STANDARD
                .decode(b64_data)
                .context("base64 decode failed")?;

            let update = AccountUpdate {
                pubkey,
                slot: params.result.context.slot,
                data,
            };

            if tx.try_send(update).is_err() {
                eprintln!("  WARNING: Update channel full, dropping update");
            }
        }

        Ok(())
    }
}
