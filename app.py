import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import timesfm
import os
from datetime import datetime

# ஹிஸ்டரியைச் சேமிப்பதற்கான ஃபைல் பெயர்
HISTORY_FILE = "trading_forecast_history.csv"

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

        column_to_forecast = st.selectbox("எந்த காலத்தை கணிப்பீட்டிற்கு பயன்படுத்த வேண்டும்?", raw_data.columns)
        forecast_length = st.slider("எத்தனை கேண்டில்களை கணிக்க வேண்டும்? (Horizon):", min_value=10, max_value=100, value=20, step=10)

        if st.button("கணிப்பை தொடங்குக (Run Forecast)"):
            with st.spinner('TimesFM மாடல் தரவை பகுப்பாய்வு செய்கிறது...'):
                try:
                    data = raw_data.dropna(subset=[column_to_forecast]).copy()
                    
                    time_col = None
                    for col in data.columns:
                        if col.lower() in ['time', 'date', 'datetime']:
                            time_col = col
                            break

                    full_prices = data[column_to_forecast].values
                    
                    # TimesFM செட்டப்
                    hparams = timesfm.TimesFmHparams(context_len=512, horizon_len=forecast_length, backend="cpu")
                    checkpoint = timesfm.TimesFmCheckpoint(huggingface_repo_id="google/timesfm-1.0-200m-pytorch")
                    tfm = timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint)
                    
                    # கணிப்பு
                    point_forecast, _ = tfm.forecast([full_prices], freq=[0]) 
                    forecast_values = point_forecast[0]
                    
                    st.success('கணிப்பு வெற்றிகரமாக முடிந்தது! 🎉')
                    
                    # --- தானியங்கி கணிப்பு விளக்கம் (Interpretation Logic) ---
                    last_historical_price = full_prices[-1]
                    final_forecast_price = forecast_values[-1]
                    pct_change = ((final_forecast_price - last_historical_price) / last_historical_price) * 100
                    
                    if pct_change > 0.5:
                        interpretation = "BULLISH (Upward Trend Momentum)"
                        alert_color = "success"
                        box_color = "#d4edda"
                    elif pct_change < -0.5:
                        interpretation = "BEARISH (Downward Selling Pressure)"
                        alert_color = "error"
                        box_color = "#f8d7da"
                    else:
                        interpretation = "SIDEWAYS (Consolidation / Rangebound)"
                        alert_color = "warning"
                        box_color = "#fff3cd"
                    
                    # ஸ்கிரீனில் தகவலைக் காட்டுதல்
                    st.markdown(f"### **AI Interpretation View:**")
                    if alert_color == "success": st.success(f"📈 **{interpretation}** | எதிர்பார்க்கப்படும் மாற்றம்: **+{pct_change:.2f}%**")
                    elif alert_color == "error": st.error(f"📉 **{interpretation}** | எதிர்பார்க்கப்படும் மாற்றம்: **{pct_change:.2f}%**")
                    else: st.warning(f"⚖️ **{interpretation}** | எதிர்பார்க்கப்படும் மாற்றம்: **{pct_change:.2f}%**")
                    
                    # --- டைம்லைன் உருவாக்கம் ---
                    if time_col:
                        data[time_col] = pd.to_datetime(data[time_col], errors='coerce')
                        data = data.dropna(subset=[time_col]) 
                        last_50_data = data.tail(50)
                        plot_historical_prices = last_50_data[column_to_forecast].values
                        historical_times = last_50_data[time_col].dt.strftime("%d-%b %H:%M").tolist()
                        
                        if len(data) >= 2: time_interval = data[time_col].iloc[-1] - data[time_col].iloc[-2]
                        else: time_interval = pd.Timedelta(minutes=5) 
                        
                        last_known_time = data[time_col].iloc[-1]
                        future_times = []
                        for i in range(1, forecast_length + 1):
                            next_time = last_known_time + (time_interval * i)
                            future_times.append(next_time.strftime("%d-%b %H:%M"))
                            
                        all_times_labels = historical_times + future_times
                    else:
                        last_50_data = data.tail(50)
                        plot_historical_prices = last_50_data[column_to_forecast].values
                        all_times_labels = [str(i) for i in range(50 + forecast_length)]
                    
                    # --- சார்ட் வரைதல் ---
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(range(50), plot_historical_prices, label="Historical Data (Last 50)", color="blue", linewidth=2)
                    
                    connected_x_values = list(range(49, 50 + forecast_length))
                    connected_y_values = [plot_historical_prices[-1]] + list(forecast_values)
                    ax.plot(connected_x_values, connected_y_values, label=f"TimesFM Forecast (Next {forecast_length})", color='red', linestyle='dashed', marker='o', markersize=4)
                    
                    # 🌟 சார்ட்டின் உள்ளே பாக்ஸ் வைக்கும் மேஜிக் (Chart Interpretation Box)
                    info_text = f"AI View: {interpretation}\nLast Close: {last_historical_price:.2f}\nProj. Target: {final_forecast_price:.2f}\nEst. Change: {pct_change:.2f}%"
                    ax.text(0.02, 0.95, info_text, transform=ax.transAxes, verticalalignment='top', fontsize=10, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.5', facecolor=box_color, alpha=0.9, edgecolor='gray'))
                    
                    tick_spacing = max(1, len(all_times_labels) // 10)
                    tick_positions = range(0, len(all_times_labels), tick_spacing)
                    tick_labels = [all_times_labels[i] for i in tick_positions]
                    ax.set_xticks(tick_positions)
                    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=9)
                    
                    ax.set_title(f"{column_to_forecast} - Future Forecast with Interpretation", fontsize=14, fontweight='bold')
                    ax.set_xlabel("Timeline (Date & Time)")
                    ax.set_ylabel("Price")
                    ax.legend(loc="lower left")
                    ax.grid(True, linestyle='--', alpha=0.6)
                    fig.tight_layout()
                    
                    st.pyplot(fig)
                    
                    # 🌟 ஹிஸ்டரியை CSV ஃபைலில் சேமிக்கும் பகுதி (Save History to File)
                    history_entry = pd.DataFrame([{
                        "Run_Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Asset_Column": column_to_forecast,
                        "Last_Close": round(last_historical_price, 2),
                        "AI_Target_Price": round(final_forecast_price, 2),
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
    st.write("நீங்கள் செய்த முந்தைய கணிப்புகளின் விவரங்கள் மற்றும் முடிவுகள் கீழே பட்டியலிடப்பட்டுள்ளன:")
    
    if os.path.isfile(HISTORY_FILE):
        history_df = pd.read_csv(HISTORY_FILE)
        # புதிய கணிப்புகள் மேலே தெரியும்படி தலைகீழாக மாற்றுதல் (Latest First)
        st.dataframe(history_df.iloc[::-1], use_container_width=True)
        
        # ஹிஸ்டரியை கிளியர் செய்ய ஒரு பட்டன்
        if st.button("வரலாற்றை அழிக்கவும் (Clear History Logs)"):
            os.remove(HISTORY_FILE)
            st.success("அனைத்து வரலாற்றுப் பதிவுகளும் வெற்றிகரமாக அழிக்கப்பட்டன! பக்கத்தை புதுப்பிக்கவும்.")
    else:
        st.info("இன்னும் எந்த கணிப்பு வரலாறும் சேமிக்கப்படவில்லை. புதிய கணிப்பைத் தொடங்கும்போது அது இங்கே பதிவாகும்!")
