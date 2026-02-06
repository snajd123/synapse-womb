//! Mock data generator: Creates a synthetic market_vibrations.bin
//! with known price movements for testing the Scanning Swarm.
//!
//! Usage: mockgen --output mock_vibrations.bin --records 10000

use anyhow::Result;
use clap::Parser;
use rand::Rng;
use solana_predator::record::{Record, write_record, AMM_DATA_SIZE};
use std::fs::File;
use std::io::BufWriter;

#[derive(Parser)]
#[command(name = "mockgen", about = "Generate synthetic market data")]
struct Args {
    /// Output file path
    #[arg(long, short, default_value = "mock_vibrations.bin")]
    output: String,

    /// Number of records to generate
    #[arg(long, short, default_value = "10000")]
    records: usize,

    /// Byte offset where the "price signal" is embedded in amm_data.
    /// Spores at this offset should converge; others should fail.
    #[arg(long, default_value = "42")]
    signal_offset: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut rng = rand::thread_rng();

    eprintln!("Generating {} mock records...", args.records);
    eprintln!("Signal embedded at byte offset: {}", args.signal_offset);

    let file = File::create(&args.output)?;
    let mut writer = BufWriter::new(file);

    // Simulate a random walk price
    let price_coin: u64 = 50_000_000_000_000; // ~50k SOL
    let mut price_pc: u64 = 7_500_000_000_000;    // ~7.5M USDC (SOL @ $150)

    // Pre-decide the NEXT direction so record i's signal byte predicts record i+1's move
    let mut next_direction: bool = rng.gen();

    for i in 0..args.records {
        let slot = 200_000_000 + i as u64;

        // This record's price move was decided in the PREVIOUS iteration's next_direction
        let direction = next_direction;
        let magnitude: u64 = rng.gen_range(1_000_000..100_000_000); // Small moves

        if direction {
            // Price up: more USDC per SOL → pc increases
            price_pc += magnitude;
        } else {
            // Price down
            if price_pc > magnitude {
                price_pc -= magnitude;
            }
        }

        // Generate mostly random AMM data
        let mut amm_data = [0u8; AMM_DATA_SIZE];
        rng.fill(&mut amm_data[..]);

        // Decide the NEXT record's direction and embed it in the MSB of the signal byte.
        // Other bits stay random, giving the Spore varied inputs with one learnable bit.
        next_direction = rng.gen();
        if next_direction {
            amm_data[args.signal_offset] |= 0x80; // Set bit 7
        } else {
            amm_data[args.signal_offset] &= 0x7F; // Clear bit 7
        }

        let record = Record {
            slot,
            amm_data,
            coin_amount: price_coin,
            pc_amount: price_pc,
        };

        write_record(&mut writer, &record)?;
    }

    eprintln!("Done. Written to: {}", args.output);
    eprintln!("Use `reader --input {}` to inspect", args.output);
    Ok(())
}
