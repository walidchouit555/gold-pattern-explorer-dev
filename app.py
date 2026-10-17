import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Gold Pattern Explorer", layout="wide")

st.title("🟠 Gold Pattern Explorer")
st.write("Historical Gold Pattern Matching & Backtesting")

st.info("النسخة الأولى: سنبدأ باستيراد البيانات واختيار العينة المستهدفة.")

st.header("1. Historical Data")
history_file = st.file_uploader(
    "Upload historical gold data (CSV)",
    type=["csv"],
    key="history"
)

if history_file is not None:
    try:
        history = pd.read_csv(history_file)
        history.columns = [c.strip().lower() for c in history.columns]

        required = ["timestamp", "open", "high", "low", "close"]
        missing = [c for c in required if c not in history.columns]

        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        else:
            history["timestamp"] = pd.to_datetime(
                history["timestamp"], utc=True, errors="coerce"
            )

            for col in ["open", "high", "low", "close"]:
                history[col] = pd.to_numeric(history[col], errors="coerce")

            history = history.dropna(
                subset=["timestamp", "open", "high", "low", "close"]
            )
            history = history.sort_values("timestamp")
            history = history.drop_duplicates("timestamp")

            st.success(f"Loaded {len(history)} valid candles")
            st.write(
                "Historical range:",
                history["timestamp"].min(),
                "—",
                history["timestamp"].max()
            )
            st.dataframe(history.tail(10), use_container_width=True)

            st.session_state["history_data"] = history

    except Exception as e:
        st.error(f"Could not read CSV: {e}")

st.divider()
st.header("2. Target Sample")

st.write(
    "Choose a historical period to use as the target, "
    "or enter 48 candles manually in a later version."
)

if "history_data" in st.session_state:
    history = st.session_state["history_data"]

    if len(history) >= 48:
        start_index = st.number_input(
            "Starting candle index",
            min_value=0,
            max_value=len(history) - 48,
            value=max(0, len(history) - 48),
            step=1
        )

        target = history.iloc[int(start_index):int(start_index) + 48]

        st.write(
            "Target period:",
            target["timestamp"].iloc[0],
            "—",
            target["timestamp"].iloc[-1]
        )

        st.line_chart(target.set_index("timestamp")["close"])

        st.caption(
            "The selected target contains 48 consecutive rows. "
            "Hourly continuity will be checked in the matching engine."
        )
    else:
        st.warning("At least 48 valid candles are required.")

st.divider()
st.header("3. Historical Matching")
st.info("The matching engine and backtesting will be added after this initial version is saved.")

# ==================================================
# 3. Historical Matching Engine
# ==================================================

st.divider()
st.header("3. Historical Matching")

st.write(
    "Compare a selected target with earlier historical "
    "patterns and inspect what happened afterward."
)

pattern_length = st.selectbox(
    "Pattern length (hourly candles)",
    [48, 72, 96, 144],
    index=3,
    key="matching_pattern_length"
)

if "history_data" in st.session_state:
    history = st.session_state["history_data"].copy()

    if len(history) >= pattern_length * 2 + 1:
        max_start = len(history) - pattern_length

        target_start = st.number_input(
            "Target starting candle index",
            min_value=0,
            max_value=max_start,
            value=max_start,
            step=1,
            key="matching_target_start"
        )

        target_start = int(target_start)
        target_end = target_start + pattern_length

        target = history.iloc[target_start:target_end].copy()

        st.write(
            "Selected target:",
            str(target["timestamp"].iloc[0]),
            "to",
            str(target["timestamp"].iloc[-1])
        )

        if len(target) == pattern_length:
            target_close = target["close"].to_numpy(dtype=float)

            # Normalize each pattern to compare its shape,
            # rather than its absolute gold price.
            target_base = target_close[0]

            if target_base != 0:
                target_shape = target_close / target_base - 1

                results = []

                                for start in range(0, target_start - pattern_length + 1):
                    end = start + pattern_length
                    candidate = history.iloc[start:end]

                    if len(candidate) != pattern_length:
                        continue

                    # Do not compare windows with missing hourly bars.
                    # Check timestamp order and duplicates.
                    times = candidate["timestamp"]

                    if not times.is_monotonic_increasing:
                        continue

                    if times.duplicated().any():
                        continue

                    candidate_close = candidate["close"].to_numpy(
                        dtype=float
                    )

                    if candidate_close[0] == 0:
                        continue

                    candidate_shape = (
                        candidate_close / candidate_close[0] - 1
                    )

                    rmse = float(
                        np.sqrt(
                            np.mean(
                                (target_shape - candidate_shape) ** 2
                            )
                        )
                    )

                    similarity = 100 / (1 + rmse * 100)

                    # Capture up to 12 later candles.
                    future = history.iloc[end:end + 12].copy()

                    results.append({
                        "start": start,
                        "end": end,
                        "rmse": rmse,
                        "similarity": similarity,
                        "future": future
                    })

                results.sort(
                    key=lambda item: item["rmse"]
                )

                st.subheader("Historical matching results")

                if not results:
                    st.warning(
                        "No eligible historical matches were found. "
                        "Check the data and selected target."
                    )
                else:
                    st.success(
                        f"Found {len(results)} historical matches."
                    )

                    top_n = st.slider(
                        "Number of matches to display",
                        min_value=1,
                        max_value=min(20, len(results)),
                        value=min(10, len(results))
                    )

                    summary = []

                    for rank, result in enumerate(
                        results[:top_n], start=1
                    ):
                        matched = history.iloc[
                            result["start"]:result["end"]
                        ]

                        summary.append({
                            "Rank": rank,
                            "Similarity (%)": round(
                                result["similarity"], 2
                            ),
                            "Start (UTC)": str(
                                matched["timestamp"].iloc[0]
                            ),
                            "End (UTC)": str(
                                matched["timestamp"].iloc[-1]
                            )
                        })

                    st.dataframe(
                        pd.DataFrame(summary),
                        use_container_width=True,
                        hide_index=True
                    )

                    st.caption(
                        "Similarity is a mathematical shape score, "
                        "not a probability or a price forecast."
                    )

                    selected_rank = st.selectbox(
                        "Inspect a historical match",
                        options=list(range(1, top_n + 1)),
                        format_func=lambda rank: f"Match {rank}"
                    )

                    selected = results[selected_rank - 1]

                    matched = history.iloc[
                        selected["start"]:selected["end"]
                    ].copy()

                    future = selected["future"]

                    st.subheader("Matched historical pattern")

                    matched_chart = matched.set_index(
                        "timestamp"
                    )["close"]

                    st.line_chart(matched_chart)

                    if not future.empty:
                        st.subheader("What happened afterward")

                        future_chart = future.set_index(
                            "timestamp"
                        )["close"]

                        st.line_chart(future_chart)

                        st.dataframe(
                            future[
                                ["timestamp", "open", "high", "low", "close"]
                            ],
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.info(
                            "No subsequent candles are available "
                            "after this historical match."
                        )

                                        st.write(
                        "Similarity score:",
                        f"{selected['similarity']:.2f}%"
                    )

            else:
                st.error(
                    "The first target closing price cannot be zero."
                )

        else:
            st.warning(
                f"The selected target does not contain {pattern_length} candles."
            )

    else:
        st.warning(
            f"At least {pattern_length * 2 + 1} valid candles are needed "
            f"to compare a {pattern_length}-candle target with an earlier pattern."
        )

else:
    st.info(
        "Upload historical CSV data in the Historical Data section first."
    )

st.divider()
st.header("4. Backtesting")
st.info(
"Historical pattern matching is available above. "
"A proper walk-forward backtest with performance statistics "
"will be implemented separately."
)



