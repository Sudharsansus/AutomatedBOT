import MetaTrader5 as mt5
import pandas as pd
import time
from datetime import datetime

class MT5Integration:
    def __init__(self, login, password, server):
        self.login = login
        self.password = password
        self.server = server
        self.connected = False

    def connect(self):
        if not mt5.initialize():
            print("initialize() failed, error code =", mt5.last_error())
            return False
        
        authorized = mt5.login(self.login, self.password, self.server)
        if not authorized:
            print("Failed to connect to MT5 account #{} on server {}. Error code: {}".format(self.login, self.server, mt5.last_error()))
            mt5.shutdown()
            return False
        
        self.connected = True
        print("Connected to MT5 account #{} on server {}".format(self.login, self.server))
        return True

    def disconnect(self):
        if self.connected:
            mt5.shutdown()
            self.connected = False
            print("Disconnected from MT5.")

    def get_account_info(self):
        if not self.connected:
            print("Not connected to MT5.")
            return None
        return mt5.account_info()

    def get_market_data(self, symbol, timeframe, count):
        if not self.connected:
            print("Not connected to MT5.")
            return None
        
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None:
            print(f"Failed to get rates for {symbol}, error code: {mt5.last_error()}")
            return None
        
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df

    def place_order(self, symbol, order_type, volume, price, deviation, stop_loss, take_profit, comment):
        if not self.connected:
            print("Not connected to MT5.")
            return None

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "deviation": deviation,
            "sl": stop_loss,
            "tp": take_profit,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC, # Good till cancel
            "type_filling": mt5.ORDER_FILLING_FOC, # Fill or Kill
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Order send failed, retcode={result.retcode}")
            # Request the result as a dictionary and display it
            result_as_dict = result._asdict()
            for field in result_as_dict:
                print("   {}": {}".format(field, result_as_dict[field]))
                if field == "request":
                    request_as_dict = result_as_dict["request"]._asdict()
                    for field in request_as_dict:
                        print("       {}": {}".format(field, request_as_dict[field]))
            return None
        
        print(f"Order placed successfully: {result}")
        return result

    def get_open_positions(self):
        if not self.connected:
            print("Not connected to MT5.")
            return None
        
        positions = mt5.positions_get()
        if positions is None:
            print(f"No positions found, error code: {mt5.last_error()}")
            return []
        
        return positions

    def close_position(self, position_ticket, volume, price, deviation):
        if not self.connected:
            print("Not connected to MT5.")
            return None

        position = mt5.positions_get(ticket=position_ticket)
        if not position:
            print(f"Position {position_ticket} not found.")
            return None

        # Determine the type of order to close the position
        order_type = mt5.ORDER_TYPE_SELL if position[0].type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position[0].symbol,
            "volume": volume,
            "type": order_type,
            "position": position_ticket,
            "price": price,
            "deviation": deviation,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_RETURN,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Close order failed, retcode={result.retcode}")
            result_as_dict = result._asdict()
            for field in result_as_dict:
                print("   {}": {}".format(field, result_as_dict[field]))
                if field == "request":
                    request_as_dict = result_as_dict["request"]._asdict()
                    for field in request_as_dict:
                        print("       {}": {}".format(field, request_as_dict[field]))
            return None
        
        print(f"Position {position_ticket} closed successfully: {result}")
        return result

# Example Usage (for testing purposes, uncomment to run)
# if __name__ == "__main__":
#     # Replace with your MT5 account details (use a demo account for testing!)
#     MT5_LOGIN = 12345678 # Your MT5 account login
#     MT5_PASSWORD = "YourPassword" # Your MT5 account password
#     MT5_SERVER = "Exness-Demo" # Your MT5 server name (e.g., Exness-Trial, Exness-Real)

#     mt5_integration = MT5Integration(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER)

#     if mt5_integration.connect():
#         # Get account information
#         # account_info = mt5_integration.get_account_info()
#         # if account_info:
#         #     print("Account Balance:", account_info.balance)

#         # Get market data (e.g., EURUSD H1 data)
#         # eurusd_data = mt5_integration.get_market_data("EURUSD", mt5.TIMEFRAME_H1, 10)
#         # if eurusd_data is not None:
#         #     print("EURUSD H1 Data:")
#         #     print(eurusd_data)

#         # Place a dummy buy order (use a demo account for this!)
#         # symbol = "XAUUSD"
#         # order_type = mt5.ORDER_TYPE_BUY
#         # volume = 0.01
#         # current_price = mt5.symbol_info_tick(symbol).ask
#         # deviation = 10 # slippage in points
#         # stop_loss = current_price - 100 * mt5.symbol_info(symbol).point # Example SL
#         # take_profit = current_price + 200 * mt5.symbol_info(symbol).point # Example TP
#         # comment = "Test Buy Order"

#         # order_result = mt5_integration.place_order(symbol, order_type, volume, current_price, deviation, stop_loss, take_profit, comment)

#         # Get open positions
#         # open_positions = mt5_integration.get_open_positions()
#         # if open_positions:
#         #     print("Open Positions:")
#         #     for pos in open_positions:
#         #         print(pos)

#         mt5_integration.disconnect()


