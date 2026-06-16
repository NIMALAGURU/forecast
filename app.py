import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import timesfm
import os
from datetime import datetime, timedelta, timezone

# ஹிஸ்டரியைச் சேமிப்பதற்கான ஃபைல் பெயர்
HISTORY_FILE = "trading_forecast_history.csv"

# இந்திய நேரத்தை (IST) செட் செய்தல்
IST = timezone(timedelta(hours=5, minutes=30))

# 🌟 CRASH-PROOF CACHE ENGINE: Removed argument to freeze memory usage at exactly 1 instance
@st.cache_resource
def load_timesfm_model():
    # Hardcoding max horizon to 100 so it handles any slider value up to 100 safely without reloading
    hparams = timesfm.TimesFmHparams(context_len=512, horizon_len=100, backend="cpu")
    checkpoint = timesfm.TimesFmCheckpoint(huggingface_repo_id="google/timesfm-1.0-200m-pytorch")
    tfm = timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint)
    return tfm

# 1. பக்க வடிவமைப்பு மற்றும் டேப்கள் (Tabs Setup)
st.set_page_config(page_title="TimesFM Trading Forecaster", layout="wide")
st.title("TimesFM Trading Forecaster Ultra Pro 📈")

tab1, tab2 = st.tabs(["📊 கணிப்பு டேஷ்போர்டு (Run Forecast)", "📜 முந்தைய கணிப்புகள் (History Log)"])

# ----------------- TAB 1: FORECAST DASHBOARD -----------------
with tab1:
    st.write("உங்கள் வரலாற்று டேட்டா CSV ஃபைலை அப்லோட் செய்து, ஏஐ கணிப்பு மற்றும் விளக்கத்தைக் காணுங்கள்.")
    uploaded_file = st.file_uploader("உங்கள் வரலாற்று டேட்டா CSV ஃபைலை அப்லோட் செய்யவும்", type=['csv'], key="uploader")

    if uploaded_file is not None:
        raw_data = pd.read_csv(uploaded_file)
        
        st.subheader("டேட்டா பிரிவியூ (Data Preview)")
        st.dataframe(raw_data.tail()) 

        # Auto-detect stock name from file name
        default_stock_name = uploaded_file.name.replace('.csv', '').replace('.CSV', '')
        stock_name = st.text_input("ஸ்டாக்கின் பெயர் (Stock Name):", value=default_stock_name)

        # வெறும் எண்கள் உள்ள காலம்களை மட்டுமே வடிகட்டி எடுத்தல்
        numeric_cols = raw_data.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_cols:
            st.error("உங்கள் CSV ஃபைலில் எண்கள் உள்ள காலம்கள் எதுவும் இல்லை!")
        else:
            default_idx = numeric_cols.index('close') if 'close' in numeric_cols else 0
            column_to_forecast = st.selectbox("எந்த காலத்தை கணிப்பீட்டிற்கு பயன்படுத்த வேண்டும்?", numeric_cols, index=default_idx)
            
            forecast_length = st.slider("எத்தனை கேண்டில்களை கணிக்க வேண்டும்? (Horizon):", min_value=10, max_value=100, value=20, step=10)

            if st.button("கணிப்பை தொடங்குக (Run Forecast)"):
                with st.spinner('TimesFM மாடல் தரவை பகுப்பாய்வு செய்கிறது... தயவுசெய்து காத்திருக்கவும்...'):
                    try:
                        predicted_time_str = datetime.now(IST).strftime("%d-%b %I:%M %p")
                        data = raw_data.dropna(subset=[column_to_forecast]).copy()
                        
                        time_col = None
                        for col in data.columns:
                            if col.lower() in ['time', 'date', 'datetime']:
                                time_col = col
                                break

                        full_prices = data[column_to_forecast].values
                        
                        # Crash-proof மாடலை அழைத்தல்
                        tfm = load_timesfm_model()
                        
                        # கணிப்பு உருவாக்குதல்
                        point_forecast, _ = tfm.forecast([full_prices], freq=[0]) 
                        forecast_values = point_forecast[0][:forecast_length] # Slice exactly to the slider length
                        
                        st.success('கணிப்பு வெற்றிகரமாக முடிந்தது! 🎉')
                        
                        # --- தானியங்கி கணிப்பு விளக்கம் (Interpretation Logic) ---
                        last_historical_price = full_prices[-1]
                        final_forecast_price = forecast_values[-1]
                        pct_change = ((final_forecast_price - last_historical_price) / last_historical_price) * 100
                        
                        if pct_change > 0.5:
                            interpretation = "BULLISH (Upward Trend)"
                            box_color = "#d4edda"
                        elif pct_change < -0.5:
                            interpretation = "BEARISH (Downward Trend)"
                            box_color = "#f8d7da"
                        else:
                            interpretation = "SIDEWAYS (Consolidation)"
                            box_color = "#fff3cd"
                        
                        # --- 🌟 NSE MARKET HOURS FILTER TIMELINE LOGIC 🌟 ---
                        target_time_str = f"T+{forecast_length}" 
                        
                        if time_col:
                            data[time_col] = pd.to_datetime(data[time_col], errors='coerce')
                            data = data.dropna(subset=[time_col]) 
                            last_50_data = data.tail(50)
                            plot_historical_prices = last_50_data[column_to_forecast].values
                            historical_times = last_50_data[time_col].dt.strftime("%d-%b %H:%M").tolist()
                            
                            if len(data) >= 2: 
                                time_interval = data[time_col].iloc[-1] - data[time_col].iloc[-2]
                            else: 
                                time_interval = pd.Timedelta(minutes=5) 
                            
                            last_known_time = data[time_col].iloc[-1]
                            future_times = []
                            current_time = last_known_time
                            
                            for _ in range(forecast_length):
                                current_time += time_interval
                                
                                # 1. மாலை 3:30 (15:30) தாண்டினால் அடுத்த நாள் காலை 9:15-க்கு மாற்றுதல்
                                if current_time.hour > 15 or (current_time.hour == 15 and current_time.minute > 30):
                                    current_time += timedelta(days=1)
                                    current_time = current_time.replace(hour=9, minute=15)
                                    
                                # 2. ஒருவேளை காலை 9:15-க்கு முன் இருந்தால் 9:15 ஆக மாற்றுதல்
                                if current_time.hour < 9 or (current_time.hour == 9 and current_time.minute < 15):
                                    current_time = current_time.replace(hour=9, minute=15)
                                    
                                # 3. சனிக்கிழமை (5) அல்லது ஞாயிற்றுக்கிழமை (6) வந்தால் திங்கட்கிழமைக்கு மாற்றுதல்
                                while current_time.weekday() >= 5:
                                    current_time += timedelta(days=1)
                                    current_time = current_time.replace(hour=9, minute=15)
                                    
                                future_times.append(current_time.strftime("%d-%b %I:%M %p"))
                                
                            all_times_labels = [pd.to_datetime(t).strftime("%d-%b %H:%M") for t in last_50_data[time_col]] + [pd.to_datetime(t).strftime("%d-%b %H:%M") for t in future_times]
                            target_time_str = future_times[-1] 
                        else:
                            last_50_data = data.tail(50)
                            plot_historical_prices = last_50_data[column_to_forecast].values
                            all_times_labels = [str(i) for i in range(50 + forecast_length)]
                        
                        # முக்கிய சிக்னல்
                        st.markdown(f"### **AI Interpretation View:**")
                        st.info(f"🔮 **Stock:** {stock_name.upper()} | **Predicted At (IST):** {predicted_time_str} | **Target Time:** {target_time_str}")
                        
                        # --- சார்ட் வரைதல் ---
                        fig, ax = plt.subplots(figsize=(12, 6))
                        ax.plot(range(50), plot_historical_prices, label="Historical Data (Last 50)", color="blue", linewidth=2)
                        
                        connected_x_values = list(range(49, 50 + forecast_length))
                        connected_y_values = [plot_historical_prices[-1]] + list(forecast_values)
                        ax.plot(connected_x_values, connected_y_values, label=f"TimesFM Forecast (Next {forecast_length})", color='red', linestyle='dashed', marker='o', markersize=4)
                        
                        # சார்ட்டின் உள்ளே இருக்கும் பாக்ஸ்
                        info_text = (
                            f"Stock: {stock_name.upper()}\n"
                            f"Predicted (IST): {predicted_time_str}\n"
                            f"AI View: {interpretation}\n"
                            f"Last Close: {last_historical_price:.2f}\n"
                            f"Proj. Target: {final_forecast_price:.2f}\n"
                            f"Target Time: {target_time_str}\n"
                            f"Est. Change: {pct_change:.2f}%"
                        )
                        ax.text(0.02, 0.95, info_text, transform=ax.transAxes, verticalalignment='top', fontsize=10, fontweight='bold',
                                bbox=dict(boxstyle='round,pad=0.5', facecolor=box_color, alpha=0.9, edgecolor='gray'))
                        
                        tick_spacing = max(1, len(all_times_labels) // 10)
                        tick_positions = range(0, len(all_times_labels), tick_spacing)
                        tick_labels = [all_times_labels[i] for i in tick_positions]
                        ax.set_xticks(tick_positions)
                        ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=9)
                        
                        ax.set_title(f"{stock_name.upper()} ({column_to_forecast}) - AI Future Forecast (NSE Market Hours Enabled)", fontsize=14, fontweight='bold')
                        ax.set_xlabel("Timeline (Date & Time)")
                        ax.set_ylabel("Price")
                        ax.legend(loc="lower left")
                        ax.grid(True, linestyle='--', alpha=0.6)
                        fig.tight_layout()
                        
                        st.pyplot(fig)
                        
                        # ஹிஸ்டரி லாக் சேமித்தல்
                        history_entry = pd.DataFrame([{
                            "Run_Timestamp": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
                            "Stock_Name": stock_name.upper(),
                            "Predicted_At_IST": predicted_time_str,
                            "Asset_Column": column_to_forecast,
                            "Last_Close": round(last_historical_price, 2),
                            "AI_Target_Price": round(final_forecast_price, 2),
                            "Target_Timeline": target_time_str,
                            "Expected_Change_Pct": round(pct_change, 2),
                            "AI_Signal": interpretation,
                            "Horizon_Candles": forecast_length
                        }])
                        
                        if not os.path.isfile(HISTORY_FILE):
                            history_entry.to_csv(HISTORY_FILE, index=False)
                        else:
                            history_entry.to_csv(HISTORY_FILE, mode='a', header=False, index=False)
                            
                    except Exception as e:
                        st.error(f"ஒரு பிழை ஏற்பட்டுள்ளது: {e}")

# ----------------- TAB 2: HISTORY LOG VIEW -----------------
with tab2:
    st.subheader("📜 சேமிக்கப்பட்ட வரலாற்று கணிப்புகள் (Saved Predictions Log)")
    if os.path.isfile(HISTORY_FILE):
        history_df = pd.read_csv(HISTORY_FILE)
        st.dataframe(history_df.iloc[::-1], use_container_width=True)
        
        if st.button("வரலாற்றை அழிக்கவும் (Clear History Logs)"):
            os.remove(HISTORY_FILE)
            st.success("அனைத்து வரலாற்றுப் பதிவுகளும் வெற்றிகரமாக அழிக்கப்பட்டன! பக்கத்தை புதுப்பிக்கவும்.")
    else:
        st.info("இன்னும் எந்த கணிப்பு வரலாறும் சேமிக்கப்படவில்லை.")
