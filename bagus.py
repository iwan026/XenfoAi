import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas_ta as ta
import MetaTrader5 as mt5
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Constants
MODEL_PATH = "forex_prediction_model.h5"
SEQUENCE_LENGTH = 60  # Number of candles to use for prediction
TIMEFRAMES = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
    "W1": mt5.TIMEFRAME_W1,
}


class ForexPredictor:
    """Class for handling all forex prediction operations"""

    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_mt5_initialized = False
        self.initialize_mt5()
        self.load_or_create_model()

    def initialize_mt5(self):
        """Initialize connection with MetaTrader 5"""
        try:
            if not mt5.initialize():
                logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False

            logger.info("MT5 initialized successfully")
            self.is_mt5_initialized = True
            return True
        except Exception as e:
            logger.error(f"Error initializing MT5: {e}")
            return False

    def load_or_create_model(self):
        """Load existing model or create a new one if it doesn't exist"""
        try:
            if os.path.exists(MODEL_PATH):
                logger.info("Loading existing model...")
                self.model = tf.keras.models.load_model(MODEL_PATH)
                logger.info("Model loaded successfully")
            else:
                logger.info("No existing model found. Creating new model...")
                self.create_model()
                logger.info("New model created successfully")
        except Exception as e:
            logger.error(f"Error loading/creating model: {e}")
            self.create_model()

    def create_model(self):
        """Create and compile the LSTM model"""
        try:
            # Input shape: (sequence_length, features)
            input_features = 15  # OHLCV + technical indicators

            self.model = tf.keras.Sequential(
                [
                    tf.keras.layers.LSTM(
                        100,
                        return_sequences=True,
                        input_shape=(SEQUENCE_LENGTH, input_features),
                    ),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.LSTM(100, return_sequences=False),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.Dense(50, activation="relu"),
                    tf.keras.layers.Dense(
                        1, activation="sigmoid"
                    ),  # Binary output (Buy/Sell)
                ]
            )

            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )
        except Exception as e:
            logger.error(f"Error creating model: {e}")

    def get_ohlc_data(self, symbol, timeframe, num_candles=5000):
        """Get OHLC data from MetaTrader 5"""
        if not self.is_mt5_initialized:
            if not self.initialize_mt5():
                return None

        try:
            # Convert timeframe string to MT5 timeframe
            mt5_timeframe = TIMEFRAMES.get(timeframe, mt5.TIMEFRAME_H1)

            # Fetch rates
            rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, num_candles)

            if rates is None or len(rates) == 0:
                return None

            # Convert to DataFrame
            df = pd.DataFrame(rates)

            # Convert timestamp to datetime
            df["time"] = pd.to_datetime(df["time"], unit="s")

            # Rename columns for clarity
            df.rename(
                columns={
                    "time": "datetime",
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "tick_volume": "volume",
                },
                inplace=True,
            )

            return df
        except Exception as e:
            logger.error(f"Error getting OHLC data: {e}")
            return None

    def load_csv_data(self, file_path):
        """Load OHLC data from CSV file"""
        try:
            df = pd.read_csv(file_path)

            # Ensure the CSV has the necessary columns
            required_columns = ["datetime", "open", "high", "low", "close", "volume"]

            # Check if datetime column exists, if not try to create it
            if "datetime" not in df.columns:
                if "time" in df.columns:
                    df.rename(columns={"time": "datetime"}, inplace=True)
                elif "date" in df.columns:
                    df.rename(columns={"date": "datetime"}, inplace=True)

            # Convert datetime to proper format if it's not already
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])

            # Check if all required columns are present
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing columns in CSV: {missing_columns}")
                return None

            return df
        except Exception as e:
            logger.error(f"Error loading CSV data: {e}")
            return None

    def add_technical_indicators(self, df):
        """Add technical indicators to the dataframe"""
        try:
            # Add RSI
            df["rsi"] = ta.rsi(df["close"], length=14)

            # Add MACD
            macd = ta.macd(df["close"])
            df = pd.concat([df, macd], axis=1)

            # Add Bollinger Bands
            bbands = ta.bbands(df["close"])
            df = pd.concat([df, bbands], axis=1)

            # Add SMA
            df["sma_20"] = ta.sma(df["close"], length=20)
            df["sma_50"] = ta.sma(df["close"], length=50)

            # Add EMA
            df["ema_9"] = ta.ema(df["close"], length=9)
            df["ema_21"] = ta.ema(df["close"], length=21)

            # Add Stochastic Oscillator
            stoch = ta.stoch(df["high"], df["low"], df["close"])
            df = pd.concat([df, stoch], axis=1)

            # Fill NaN values with forward fill then backward fill
            df.fillna(method="ffill", inplace=True)
            df.fillna(method="bfill", inplace=True)

            return df
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            return df

    def prepare_data_for_training(self, df):
        """Prepare data for training the model"""
        try:
            # Add technical indicators
            df = self.add_technical_indicators(df)

            # Create target variable (1 for price increase, 0 for decrease)
            df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

            # Drop rows with NaN values
            df.dropna(inplace=True)

            # Select features for training
            feature_columns = [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "rsi",
                "MACD_12_26_9",
                "MACDs_12_26_9",
                "BBL_5_2.0",
                "BBM_5_2.0",
                "BBU_5_2.0",
                "sma_20",
                "sma_50",
                "ema_9",
                "ema_21",
            ]

            # Create sequences for LSTM
            X, y = [], []
            for i in range(len(df) - SEQUENCE_LENGTH):
                X.append(df[feature_columns].iloc[i : i + SEQUENCE_LENGTH].values)
                y.append(df["target"].iloc[i + SEQUENCE_LENGTH])

            X = np.array(X)
            y = np.array(y)

            # Normalize features
            X_reshaped = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
            X_normalized = self.scaler.fit_transform(X_reshaped)
            X = X_normalized.reshape(X.shape)

            return X, y
        except Exception as e:
            logger.error(f"Error preparing data for training: {e}")
            return None, None

    def train_model(self, file_path=None, df=None, epochs=50, batch_size=32):
        """Train the model with data from CSV or DataFrame"""
        try:
            if df is None and file_path is not None:
                df = self.load_csv_data(file_path)

            if df is None:
                logger.error("No data provided for training")
                return False

            X, y = self.prepare_data_for_training(df)

            if X is None or y is None:
                logger.error("Failed to prepare data for training")
                return False

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Train model
            self.model.fit(
                X_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                verbose=1,
            )

            # Evaluate model
            loss, accuracy = self.model.evaluate(X_test, y_test)
            logger.info(f"Model evaluation - Loss: {loss}, Accuracy: {accuracy}")

            # Save the model
            self.model.save(MODEL_PATH)
            logger.info(f"Model saved to {MODEL_PATH}")

            return True
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False

    def prepare_data_for_prediction(self, df):
        """Prepare data for making predictions"""
        try:
            # Add technical indicators
            df = self.add_technical_indicators(df)

            # Select features for prediction
            feature_columns = [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "rsi",
                "MACD_12_26_9",
                "MACDs_12_26_9",
                "BBL_5_2.0",
                "BBM_5_2.0",
                "BBU_5_2.0",
                "sma_20",
                "sma_50",
                "ema_9",
                "ema_21",
            ]

            # Ensure we have enough data
            if len(df) < SEQUENCE_LENGTH:
                logger.error(
                    f"Not enough data for prediction. Need at least {SEQUENCE_LENGTH} candles."
                )
                return None

            # Get the last sequence_length rows
            df_sequence = df[feature_columns].iloc[-SEQUENCE_LENGTH:].values

            # Reshape and normalize
            df_sequence_reshaped = df_sequence.reshape(
                1, df_sequence.shape[0] * df_sequence.shape[1]
            )
            df_sequence_normalized = self.scaler.transform(df_sequence_reshaped)
            df_sequence = df_sequence_normalized.reshape(
                1, SEQUENCE_LENGTH, len(feature_columns)
            )

            return df_sequence
        except Exception as e:
            logger.error(f"Error preparing data for prediction: {e}")
            return None

    def predict(self, symbol, timeframe):
        """Make prediction for a symbol and timeframe"""
        try:
            if self.model is None:
                logger.error("Model not initialized")
                return None

            # Get data from MT5
            df = self.get_ohlc_data(
                symbol, timeframe, SEQUENCE_LENGTH + 100
            )  # Get extra data for indicators

            if df is None:
                logger.error(f"Failed to get data for {symbol} on {timeframe}")
                return None

            # Prepare data for prediction
            X = self.prepare_data_for_prediction(df)

            if X is None:
                return None

            # Make prediction
            prediction = self.model.predict(X)[0][0]

            # Get additional indicator signals
            last_row = df.iloc[-1]

            # RSI signal
            rsi_signal = (
                "Buy"
                if last_row["rsi"] < 30
                else "Sell"
                if last_row["rsi"] > 70
                else "Neutral"
            )

            # MACD signal
            macd_signal = (
                "Buy"
                if last_row["MACD_12_26_9"] > last_row["MACDs_12_26_9"]
                else "Sell"
            )

            # Moving Average signal
            ma_signal = "Buy" if last_row["ema_9"] > last_row["sma_20"] else "Sell"

            # Combined signal
            combined_signals = [
                "Buy" if prediction > 0.5 else "Sell",  # AI prediction
                rsi_signal,
                macd_signal,
                ma_signal,
            ]

            # Count signals
            buy_count = combined_signals.count("Buy")
            sell_count = combined_signals.count("Sell")

            # Final decision
            final_signal = "Buy" if buy_count > sell_count else "Sell"
            confidence = max(buy_count, sell_count) / len(combined_signals)

            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "prediction": "Buy" if prediction > 0.5 else "Sell",
                "confidence": float(prediction)
                if prediction > 0.5
                else 1 - float(prediction),
                "rsi_signal": rsi_signal,
                "macd_signal": macd_signal,
                "ma_signal": ma_signal,
                "final_signal": final_signal,
                "final_confidence": confidence,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return None

    def __del__(self):
        """Clean up resources"""
        try:
            if self.is_mt5_initialized:
                mt5.shutdown()
                logger.info("MT5 connection closed")
        except:
            pass


class TelegramBot:
    """Class for handling Telegram bot operations"""

    def __init__(self, token):
        """Initialize the bot with a token"""
        self.token = token
        self.forex_predictor = ForexPredictor()
        self.application = Application.builder().token(token).build()
        self.setup_handlers()

    def setup_handlers(self):
        """Set up command handlers"""
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("prediksi", self.predict_command))
        self.application.add_handler(CommandHandler("train", self.train_command))
        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message)
        )

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Send a welcome message when the /start command is issued"""
        welcome_message = (
            "Selamat datang di Bot Prediksi Forex!\n\n"
            "Perintah yang tersedia:\n"
            "/prediksi [PAIR] [TIMEFRAME] - Melakukan prediksi untuk pair dan timeframe tertentu\n"
            "Contoh: /prediksi EURUSD H1\n\n"
            "/train [FILE_PATH] - Melatih model dengan data dari file CSV\n"
            "Contoh: /train D:/data/EURUSD_H1.csv\n\n"
            "/help - Menampilkan bantuan"
        )
        await update.message.reply_text(welcome_message)

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Send a help message when the /help command is issued"""
        help_message = (
            "Cara Menggunakan Bot Prediksi Forex:\n\n"
            "1. Prediksi Forex:\n"
            "   /prediksi [PAIR] [TIMEFRAME]\n"
            "   Contoh: /prediksi EURUSD H1\n\n"
            "   Timeframe yang tersedia: M1, M5, M15, H1, H4, D1, W1\n\n"
            "2. Melatih Model:\n"
            "   /train [FILE_PATH]\n"
            "   Contoh: /train D:/data/EURUSD_H1.csv\n\n"
            "Pastikan MetaTrader 5 terhubung untuk mendapatkan data real-time."
        )
        await update.message.reply_text(help_message)

    async def predict_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle prediction command"""
        try:
            # Get arguments
            args = context.args

            if len(args) != 2:
                await update.message.reply_text(
                    "Format perintah salah. Gunakan:\n"
                    "/prediksi [PAIR] [TIMEFRAME]\n"
                    "Contoh: /prediksi EURUSD H1"
                )
                return

            symbol = args[0].upper()
            timeframe = args[1].upper()

            if timeframe not in TIMEFRAMES:
                await update.message.reply_text(
                    f"Timeframe tidak valid. Gunakan salah satu dari: {', '.join(TIMEFRAMES.keys())}"
                )
                return

            # Send processing message
            await update.message.reply_text(
                f"Memproses prediksi untuk {symbol} pada timeframe {timeframe}..."
            )

            # Get prediction
            prediction = self.forex_predictor.predict(symbol, timeframe)

            if prediction is None:
                await update.message.reply_text(
                    f"Gagal mendapatkan prediksi untuk {symbol} pada timeframe {timeframe}. "
                    "Pastikan simbol valid dan MetaTrader 5 terhubung."
                )
                return

            # Format result
            result_message = (
                f"📊 *Prediksi Forex untuk {symbol} ({timeframe})*\n\n"
                f"🔮 *Prediksi AI:* {prediction['prediction']}\n"
                f"⚖️ *Keyakinan:* {prediction['confidence']:.2f}\n\n"
                f"*Sinyal Indikator:*\n"
                f"📈 RSI: {prediction['rsi_signal']}\n"
                f"📉 MACD: {prediction['macd_signal']}\n"
                f"📊 MA: {prediction['ma_signal']}\n\n"
                f"*Keputusan Akhir:* {prediction['final_signal']} dengan keyakinan {prediction['final_confidence']:.2f}\n\n"
                f"🕒 *Waktu:* {prediction['timestamp']}"
            )

            await update.message.reply_text(result_message, parse_mode="Markdown")

            # Log prediction
            logger.info(
                f"Prediction for {symbol} ({timeframe}): {prediction['final_signal']}"
            )
        except Exception as e:
            logger.error(f"Error in predict_command: {e}")
            await update.message.reply_text(f"Terjadi kesalahan: {str(e)}")

    async def train_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle training command"""
        try:
            # Get arguments
            args = context.args

            if len(args) != 1:
                await update.message.reply_text(
                    "Format perintah salah. Gunakan:\n"
                    "/train [FILE_PATH]\n"
                    "Contoh: /train D:/data/EURUSD_H1.csv"
                )
                return

            file_path = args[0]

            # Check if file exists
            if not os.path.exists(file_path):
                await update.message.reply_text(f"File {file_path} tidak ditemukan.")
                return

            # Send processing message
            await update.message.reply_text(
                f"Melatih model dengan data dari {file_path}...\nIni akan memakan waktu beberapa menit."
            )

            # Train model
            success = self.forex_predictor.train_model(file_path=file_path)

            if success:
                await update.message.reply_text("Model berhasil dilatih dan disimpan.")
            else:
                await update.message.reply_text(
                    "Gagal melatih model. Periksa log untuk informasi lebih lanjut."
                )
        except Exception as e:
            logger.error(f"Error in train_command: {e}")
            await update.message.reply_text(f"Terjadi kesalahan: {str(e)}")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle regular messages"""
        message_text = update.message.text.lower()

        if "prediksi" in message_text:
            # Try to extract pair and timeframe
            words = message_text.split()
            if len(words) >= 3:
                for i, word in enumerate(words):
                    if word == "prediksi" and i + 2 < len(words):
                        pair = words[i + 1].upper()
                        timeframe = words[i + 2].upper()

                        if timeframe in TIMEFRAMES:
                            # Call predict command
                            context.args = [pair, timeframe]
                            await self.predict_command(update, context)
                            return

            await update.message.reply_text(
                "Format pesan salah. Gunakan:\n"
                "/prediksi [PAIR] [TIMEFRAME]\n"
                "Contoh: /prediksi EURUSD H1"
            )
        else:
            await update.message.reply_text(
                "Saya tidak mengerti pesan Anda. Gunakan /help untuk melihat perintah yang tersedia."
            )

    def run(self):
        """Run the bot"""
        logger.info("Starting bot...")
        self.application.run_polling()


def main():
    """Main function"""
    try:
        # Get token from environment variable or use default
        token = os.environ.get("TELEGRAM_TOKEN", "YOUR_TELEGRAM_TOKEN_HERE")

        # Initialize and run the bot
        bot = TelegramBot(token)
        bot.run()
    except Exception as e:
        logger.error(f"Error in main: {e}")


if __name__ == "__main__":
    main()
