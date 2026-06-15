import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import timesfm

# 1. ஆப் டைட்டில் மற்றும் வடிவமைப்பு (App Title)
st.title("TimesFM Trading Forecaster PRO 📈")
st.write("உங்கள் Bank Nifty அல்லது ஸ்டாக் டேட்டாவை (CSV) இங்கே அப்லோட் செய்து எதிர்கால விலையை கணிக்கவும்.")
st.markdown("---")

# 2. ஃபைல் அப்லோடர் (File Uploader)
uploaded_file = st.file_uploader("உங்கள் வரலாற்று டேட்டா CSV ஃபைலை அப்லோட் செய்யவும்", type=['csv'])

if uploaded_file is not None:
    raw_data = pd.read_csv(uploaded_file)
    
    st.subheader("டேட்டா பிரிவியூ (Data Preview)")
    st.dataframe(raw_data.tail()) 

    st.write("எந்த டேட்டாவை வைத்து கணிக்க வேண்டும்?")
    column_to_forecast = st.selectbox("காலத்தைத் தேர்ந்தெடுக்கவும்:", raw_data.columns)

    st.markdown("---")
    forecast_length = st.slider("எத்தனை கேண்டில்களை கணிக்க வேண்டும்? (Forecast Horizon):", min_value=10, max_value=100, value=20, step=10)

    if st.button("கணிப்பை தொடங்கு (Run Forecast)"):
        with st.spinner('TimesFM மாடல் டேட்டாவை அனலைஸ் செய்கிறது... தயவுசெய்து காத்திருக்கவும்...'):
            try:
                # Clean trailing empty/NaN rows
                data = raw_data.dropna(subset=[column_to_forecast]).copy()
                
                # Find time column
                time_col = None
                for col in data.columns:
                    if col.lower() in ['time', 'date', 'datetime']:
                        time_col = col
                        break

                full_prices = data[column_to_forecast].values
                
                # TimesFM Model Setup
                hparams = timesfm.TimesFmHparams(
                    context_len=512,      
                    horizon_len=forecast_length,       
                    backend="cpu",        
                )
                
                checkpoint = timesfm.TimesFmCheckpoint(
                    huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
                )
                
                tfm = timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint)
                
                # Run AI Forecast
                point_forecast, quantile_forecast = tfm.forecast([full_prices], freq=[0]) 
                forecast_values = point_forecast[0]
                
                st.success('கணிப்பு வெற்றிகரமாக முடிந்தது! 🎉')
                
                # Generate exact future timestamps dynamically
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
                    for i in range(1, forecast_length + 1):
                        next_time = last_known_time + (time_interval * i)
                        future_times.append(next_time.strftime("%d-%b %H:%M"))
                        
                    all_times_labels = historical_times + future_times
                else:
                    last_50_data = data.tail(50)
                    plot_historical_prices = last_50_data[column_to_forecast].values
                    all_times_labels = [str(i) for i in range(50 + forecast_length)]
                
                # 5. Plotting the Result
                st.subheader("கணிக்கப்பட்ட சார்ட் (Forecasted Chart)")
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Plot blue line for historical data
                ax.plot(range(50), plot_historical_prices, label="Historical Data (Last 50)", color="blue")
                
                # 🌟 THE FIX: Bridging the Gap 🌟
                last_historical_price = plot_historical_prices[-1]
                
                # Combine the last historical coordinate with the new forecasted coordinates
                connected_x_values = list(range(49, 50 + forecast_length))
                connected_y_values = [last_historical_price] + list(forecast_values)
                
                # Plot the connected red line
                ax.plot(connected_x_values, connected_y_values, label=f"TimesFM Forecast (Next {forecast_length})", color='red', linestyle='dashed', marker='o')
                
                # Clean X-Axis
                tick_spacing = max(1, len(all_times_labels) // 10)
                tick_positions = range(0, len(all_times_labels), tick_spacing)
                tick_labels = [all_times_labels[i] for i in tick_positions]
                
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=9)
                fig.tight_layout()
                
                ax.set_title(f"{column_to_forecast} - Future Forecast with Exact Timeline")
                ax.set_xlabel("Timeline (Date & Time)")
                ax.set_ylabel("Price")
                ax.legend()
                ax.grid(True)
                
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"ஒரு பிழை ஏற்பட்டுள்ளது: {e}")
