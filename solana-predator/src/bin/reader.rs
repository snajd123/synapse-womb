//! Reader: Validate and inspect a market_vibrations.bin file.
//!
//! Usage: reader --input market_vibrations.bin [--limit 10]

use anyhow::{Context, Result};
use clap::Parser;
use solana_predator::record::{read_record, RECORD_SIZE};
use std::fs::File;
use std::io::BufReader;

#[derive(Parser)]
#[command(name = "reader", about = "Inspect market_vibrations.bin")]
struct Args {
    /// Input file path
    #[arg(long, short, default_value = "market_vibrations.bin")]
    input: String,

    /// Max records to display (0 = all)
    #[arg(long, default_value = "20")]
    limit: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let file = File::open(&args.input)
        .with_context(|| format!("Failed to open: {}", args.input))?;
    let file_size = file.metadata()?.len();
    let expected_records = file_size / RECORD_SIZE as u64;
    let remainder = file_size % RECORD_SIZE as u64;

    eprintln!("File: {}", args.input);
    eprintln!("Size: {} bytes", file_size);
    eprintln!("Expected records: {} (remainder: {} bytes)", expected_records, remainder);
    if remainder != 0 {
        eprintln!("WARNING: File size is not a multiple of record size ({})", RECORD_SIZE);
    }
    eprintln!();

    let mut reader = BufReader::new(file);
    let mut count = 0;
    let mut prev_price = 0.0_f64;

    while let Some(record) = read_record(&mut reader)? {
        let price = record.price(9, 6);
        let direction = if count == 0 {
            " "
        } else if price > prev_price {
            "^"
        } else if price < prev_price {
            "v"
        } else {
            "="
        };

        if args.limit == 0 || count < args.limit {
            println!(
                "#{:6} | slot {:>12} | price {:>10.4} {} | coin {:>15} | pc {:>15} | data[0..4]: {:02x} {:02x} {:02x} {:02x}",
                count,
                record.slot,
                price,
                direction,
                record.coin_amount,
                record.pc_amount,
                record.amm_data[0],
                record.amm_data[1],
                record.amm_data[2],
                record.amm_data[3],
            );
        }

        prev_price = price;
        count += 1;
    }

    eprintln!("\nTotal records read: {}", count);
    Ok(())
}
