//! gRPC source using Helius Yellowstone (Geyser).
//!
//! Behind `#[cfg(feature = "grpc")]` — requires a paid Helius plan.

use anyhow::{Context, Result};
use futures::StreamExt;
use std::collections::HashMap;
use tokio::sync::mpsc;

use yellowstone_grpc_client::GeyserGrpcClient;
use yellowstone_grpc_proto::prelude::subscribe_update::UpdateOneof;
use yellowstone_grpc_proto::prelude::{
    CommitmentLevel, SubscribeRequest, SubscribeRequestFilterAccounts,
};

use super::{AccountUpdate, MarketConfig};

#[derive(Clone)]
pub struct GrpcSource;

impl GrpcSource {
    pub fn new() -> Self {
        Self
    }

    /// Connect to the gRPC endpoint, subscribe to all accounts in `config`,
    /// and push `AccountUpdate`s into `tx` until the connection drops.
    pub async fn stream(
        &self,
        config: &MarketConfig,
        tx: mpsc::Sender<AccountUpdate>,
    ) -> Result<()> {
        let mut client = GeyserGrpcClient::build_from_shared(config.endpoint.clone())
            .and_then(|b| b.x_token(config.api_key.clone()))
            .context("gRPC build failed")?
            .connect()
            .await
            .context("gRPC connect failed")?;

        let mut accounts_filter = HashMap::new();
        accounts_filter.insert(
            "pool".to_string(),
            SubscribeRequestFilterAccounts {
                account: config.accounts.clone(),
                owner: vec![],
                filters: vec![],
                nonempty_txn_signature: None,
            },
        );

        let request = SubscribeRequest {
            accounts: accounts_filter,
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
            .context("gRPC subscribe failed")?;

        while let Some(message) = stream.next().await {
            let msg = message.context("gRPC stream error")?;

            if let Some(UpdateOneof::Account(account_update)) = msg.update_oneof {
                if let Some(account_info) = &account_update.account {
                    let pubkey = bs58::encode(&account_info.pubkey).into_string();
                    let update = AccountUpdate {
                        pubkey,
                        slot: account_update.slot,
                        data: account_info.data.clone(),
                    };

                    if tx.try_send(update).is_err() {
                        eprintln!("  WARNING: Update channel full, dropping update");
                    }
                }
            }
        }

        Ok(())
    }
}
