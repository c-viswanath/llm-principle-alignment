import pandas as pd
import numpy as np
import os

# ================= CONFIGURATION =================
INPUT_CSV = "/home/viswanath/llm-alignment/spanish/spanish_results.csv" # Update to your actual results filename
OUTPUT_REPORT_CSV = "spanish_model_leaderboard.csv"
    
# CORRECTED PREFIXES based on your CSV headers
MODEL_PREFIXES = [
    "gemma2",
    "phi3",
    "yantien_gemma2_uncensored",
    "mistral",
    "deepseek_r1",
    "rolandroland_llama3_1_uncensored",
    "qwen2",
    "llama2",
    "llama3_1",
    "phi3_5",
    "phi4_mini_reasoning",
    "qwen3",
    "gpt_oss"
]

PASSES = [
    ("Base", "base_choice"),
    ("Think", "think_choice"),
    ("Principle", "principle_choice"),
    ("Principle+Think", "principle_think_choice")
]
# =================================================

def analyze_and_save():
    if not os.path.exists(INPUT_CSV):
        print(f"❌ File not found: {INPUT_CSV}")
        return

    print(f"📄 Loading: {INPUT_CSV}...")
    try:
        df = pd.read_csv(INPUT_CSV)
        # CRITICAL: Clean column names to remove trailing spaces
        df.columns = df.columns.str.strip()
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    # 1. SETUP GROUND TRUTH
    # Look for 'Ground_Truths' or variations
    gt_col = next((c for c in df.columns if c.lower() in ['ground_truths', 'ground truth', 'groundtruth']), None)
    
    if not gt_col:
        print("❌ Error: 'Ground_Truths' column missing in dataset.")
        print(f"Found columns: {list(df.columns)}")
        return

    # Normalize Ground Truths
    df[gt_col] = df[gt_col].astype(str).str.strip().str.upper()
    
    # Filter rows where Ground_Truths is strictly A or B
    valid_gt_df = df[df[gt_col].isin(['A', 'B'])].copy()
    valid_gt_count = len(valid_gt_df)
    
    print(f"✅ Analysis based on {valid_gt_count} rows with valid Ground Truths (A/B).")

    # List to store stats for final CSV
    summary_data = []

    print("=" * 120)
    print(f"{'MODEL & PASS':<55} | {'ACCURACY':<10} | {'VALIDITY':<10} | {'FAILSAFE':<10} | {'B-BIAS':<10}")
    print("=" * 120)

    for prefix in MODEL_PREFIXES:
        
        # Check if this model exists in the CSV
        # We look for the "_base_choice" column to confirm existence
        base_col = f"{prefix}_base_choice"
        if base_col not in df.columns:
            # Try finding partial match
            if not any(c.startswith(prefix) for c in df.columns):
                continue

        print(f"\n🔹 MODEL: {prefix}")
        print("-" * 120)

        for pass_label, suffix in PASSES:
            choice_col = f"{prefix}_{suffix}"
            rationale_col = choice_col.replace("choice", "rationale")
            
            if choice_col not in df.columns:
                print(f"{pass_label:<55} | ⚠️ COLUMN MISSING")
                continue

            # --- DATA PREP ---
            # Extract relevant columns from valid rows
            subset = valid_gt_df.copy()
            
            # Normalize Predictions
            subset['cleaned_pred'] = subset[choice_col].astype(str).str.strip().str.upper()
            
            # --- FAILSAFE DETECTION ---
            # 1. Empty/NaN implies a crash or no generation
            # 2. 'FAILSAFE' in rationale implies the retry logic kicked in
            is_failsafe = pd.Series([False] * len(subset), index=subset.index)
            
            if rationale_col in subset.columns:
                raw_rationale = subset[rationale_col].astype(str).str.lower()
                is_failsafe = raw_rationale.str.contains("failsafe", na=False)
            
            failsafe_count = is_failsafe.sum()
            failsafe_rate = (failsafe_count / valid_gt_count) * 100

            # --- VALIDITY ---
            # Rows where the model actually output A or B
            valid_preds_mask = subset['cleaned_pred'].isin(['A', 'B'])
            valid_response_count = valid_preds_mask.sum()
            
            # Calculate Validity % against Total Test Rows
            validity_pct = (valid_response_count / valid_gt_count) * 100

            # --- ACCURACY & BIAS ---
            if valid_response_count == 0:
                acc_val = 0
                bias_val = 0
                acc_str = "N/A"
            else:
                # Filter to only valid A/B rows for accuracy calculation
                final_calc_df = subset[valid_preds_mask]
                
                # Accuracy: (Pred == GT)
                matches = (final_calc_df['cleaned_pred'] == final_calc_df[gt_col]).sum()
                acc_val = (matches / valid_response_count) * 100
                
                # B-Bias: % of B answers
                b_count = (final_calc_df['cleaned_pred'] == 'B').sum()
                bias_val = (b_count / valid_response_count) * 100

                acc_str = f"{acc_val:.2f}%"
                if acc_val > 80: acc_str += " 🏆"
                elif acc_val < 50: acc_str += " 📉"

                # Bias warning
                if bias_val > 90 or bias_val < 10:
                    acc_str += " 🚨" # Flag collapse

            # Print
            print(f"{pass_label:<55} | {acc_str:<10} | {validity_pct:.0f}%       | {failsafe_rate:.1f}%       | {bias_val:.1f}%")

            # Store for CSV
            summary_data.append({
                "Model": prefix,
                "Pass": pass_label,
                "Accuracy": acc_val,
                "Validity": validity_pct,
                "Failsafe_Rate": failsafe_rate,
                "B_Bias": bias_val
            })

        print("-" * 120)

    # 3. SAVE REPORT
    if summary_data:
        report_df = pd.DataFrame(summary_data)
        report_df.to_csv(OUTPUT_REPORT_CSV, index=False)
        print(f"\n✅ Summary Report saved to: {OUTPUT_REPORT_CSV}")
    else:
        print("\n❌ No valid data found to summarize.")

if __name__ == "__main__":
    analyze_and_save()