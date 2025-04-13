# TQQQ 전략 시뮬레이터 (백테스트용)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from io import BytesIO
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

class TQQQSimulator:
    def __init__(self, ticker="TQQQ", start_date="2020-01-01", end_date="2024-12-31", per_buy_amount=1_000_000, buy_interval=5, initial_cash=0, signal_ticker="TQQQ", entry_drawdown=20, exit_recovery=10, stop_buy_rally=5):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.df = self.download_data()
        self.per_buy_amount = per_buy_amount
        self.buy_interval = buy_interval
        self.cash = initial_cash
        self.initial_cash = initial_cash  # 최초 보유자산 저장
        self.shares = 0
        self.portfolio = []
        self.mdd_history = []
        self.daily_value = []
        self.sell_points = []
        self.cumulative_shares = []
        self.cash_history = []
        self.cash_shortage_points = []
        self.cash_ratio_history = []
        self.signal_ticker = signal_ticker
        self.entry_drawdown = entry_drawdown
        self.exit_recovery = exit_recovery
        signal_start_date = (pd.to_datetime(self.start_date) - pd.DateOffset(years=2)).strftime('%Y-%m-%d')
        self.signal_df = self.download_data(ticker=signal_ticker, start=signal_start_date)
        self.stop_buy_rally = stop_buy_rally
        self.signal_max = self.signal_df['Close'].cummax()
        self.df, self.signal_df = self.df.align(self.signal_df, join='inner', axis=0)
        self.compute_indicators()

    def download_data(self, ticker=None, start=None):
        if ticker is None:
            ticker = self.ticker
        if start is None:
            start = self.start_date
        df = yf.download(ticker, start=start, end=self.end_date)
        df = df[['Close']].copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.dropna(inplace=True)
        return df

    def compute_indicators(self):
        self.df['60MA'] = self.df['Close'].rolling(window=60).mean()
        delta = self.df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        self.df['RSI'] = 100 - (100 / (1 + rs))
        self.df['Deviation'] = ((self.df['Close'] - self.df['60MA']) / self.df['60MA']) * 100
        self.df['Week'] = self.df.index.to_period('W')

    def get_current_mdd(self):
        if not self.daily_value:
            return 0
        values = [v['Value'] for v in self.daily_value]
        peak = max(values)
        current = values[-1]
        if peak == 0:
            return 0
        return (peak - current) / peak * 100

    def simulate(self, rebalance_interval=30, rebalance_target_stock_ratio=0.8):
        rebalance_day_counter = 0
        just_entered = False
        rebalance_day_counter = 0
        current_week = None
        last_week_rsi = None
        peak_value = 0
        max_drawdown = 0
        in_position = False
        entry_peak = None

        for i in range(len(self.df)):
            rebalance_day_counter += 1
            row = self.df.iloc[i]
            date, price, rsi, dev, week = row.name, row['Close'], row['RSI'], row['Deviation'], row['Week']

            if week != current_week:
                current_week = week
                last_week_rsi = rsi

            signal_price = self.signal_df['Close'].iloc[i]
            signal_peak = self.signal_df['Close'].loc[:date].max()
            drawdown = (signal_price - signal_peak) / signal_peak * 100

            if not in_position and drawdown <= -self.entry_drawdown and signal_price <= signal_peak * (1 + self.stop_buy_rally / 100):
                entry_peak = signal_peak
                drawdown_from_entry_peak = (signal_price - entry_peak) / entry_peak * 100
                self.buy(date, price, f'진입(DD {drawdown:.2f}%)', self.per_buy_amount, signal_peak, drawdown_from_entry_peak)
                self.portfolio[-1]['진입 시점 고점'] = signal_peak
                just_entered = True
                in_position = True

            if in_position:
                drawdown_from_entry_peak = (signal_price - entry_peak) / entry_peak * 100
            else:
                drawdown_from_entry_peak = None

            if in_position and drawdown_from_entry_peak is not None and drawdown_from_entry_peak >= self.exit_recovery:
                self.sell(date, price, f'청산(DD {drawdown_from_entry_peak:.2f}%)', self.shares)
                self.sell_points.append((date, price))
                in_position = False

            elif not in_position and signal_price >= signal_peak * (1 + self.stop_buy_rally / 100):
                self.buy(date, price, f'매수중지(고점 {self.stop_buy_rally}%)', 0, signal_peak, drawdown_from_entry_peak)

            if in_position and i % self.buy_interval == 0 and not just_entered:
                if signal_price >= entry_peak * (1 + self.stop_buy_rally / 100):
                    self.buy(date, price, f'매수중지(고점 {self.stop_buy_rally}%)', 0, entry_peak, drawdown_from_entry_peak)
                else:
                    self.buy(date, price, '정기매수', self.per_buy_amount, entry_peak, drawdown_from_entry_peak)

            portfolio_value = self.shares * price
            peak_value = max(peak_value, portfolio_value)
            if peak_value > 0:
                drawdown = (peak_value - portfolio_value) / peak_value
                max_drawdown = max(max_drawdown, drawdown)

            total_value = portfolio_value + self.cash
            self.daily_value.append({"Date": date, "Value": total_value})
            self.cash_history.append({"Date": date, "Cash": self.cash})
            if total_value > 0:
                self.cash_ratio_history.append({"Date": date, "CashRatio": round(self.cash / total_value * 100, 1), "StockRatio": round(self.shares * price / total_value * 100, 1), "TotalValue": round(total_value, 1), "Remaining Cash": round(self.cash, 1), "Stock Value": round(self.shares * price, 1)})
            else:
                self.cash_ratio_history.append({"Date": date, "CashRatio": 0})
            if self.cash <= 0:
                self.cash_shortage_points.append(date)
            self.cumulative_shares.append({"Date": date, "Shares": self.shares})

            # 수정된 리밸런싱 조건
            if rebalance_day_counter >= rebalance_interval:
                if self.shares > 0:
                    total_value = self.shares * price + self.cash
                    current_stock_value = self.shares * price
                    current_stock_ratio = current_stock_value / total_value
                     
                    print(f"{date} - 주식비중: {round(current_stock_ratio * 100, 1)}% (목표: {rebalance_target_stock_ratio}%)")

                    if current_stock_ratio > rebalance_target_stock_ratio/100:
                        target_stock_value = total_value * rebalance_target_stock_ratio/100
                        excess_value = current_stock_value - target_stock_value
                        quantity_to_sell = excess_value / price
                        self.sell(date, price, '리밸런싱 매도', quantity_to_sell)
                rebalance_day_counter = 0

            just_entered = False

        final_price = self.df['Close'].iloc[-1]
        final_value = self.shares * final_price
        total_shares_bought = sum(x['Shares Bought'] for x in self.portfolio if x['Shares Bought'] > 0)
        return {
            '최초 보유자산': self.initial_cash,
            '총 매수 수량': round(total_shares_bought, 1),
            '보유 주식 수': self.shares,
            '최종 평가금액': final_value + self.cash,
            '수익률(%)': round(100 * (final_value + self.cash - self.initial_cash) / self.initial_cash, 2),
            'MDD(%)': max([v['MDD'] for v in self.mdd_history]) if self.mdd_history else 0,
            '매수 기록': pd.DataFrame(self.portfolio),
            '자산 추이': pd.DataFrame(self.daily_value),
            '누적 주식 수': pd.DataFrame(self.cumulative_shares),
            '현금 추이': pd.DataFrame(self.cash_history),
            '현금 비중': pd.DataFrame(self.cash_ratio_history),
            '현금 부족 시점': self.cash_shortage_points,
            '총 매수 횟수': len([x for x in self.portfolio if x['Action'].startswith('정기매수') or x['Action'].startswith('추가매수')]),
            '매도 시점': self.sell_points
        }


    def buy(self, date, price, action, amount, signal_peak=None, drawdown=None):
        if self.cash <= 0:
            return
        if amount > self.cash:
            return
        quantity = amount / price
        self.cash -= amount
        self.shares += quantity
        portfolio_value = self.shares * price + self.cash
        mdd = self.get_current_mdd()
        self.mdd_history.append({'Date': date, 'MDD': mdd})
        self.portfolio.append({
            'Date': date,
            'Price': round(price, 1),
            'Action': action,
            'Amount': round(amount, 1),
            'Shares Bought': round(quantity, 1),
            '신규진입': bool('진입' in action),
            '매수중지': bool('중단' in action),
            '청산': bool('청산' in action),
            f"기준 주가({self.signal_ticker})": round(self.signal_df['Close'].loc[date], 1),
            '진입 시점 고점': round(signal_peak, 1) if signal_peak is not None else np.nan,
            'Drawdown (%)': round(drawdown, 1) if drawdown is not None else '',
            'Total MDD (%)': round(mdd, 1)
        })

    def sell(self, date, price, action, quantity):
        if quantity > self.shares:
            quantity = self.shares
        amount = quantity * price
        self.cash += amount
        self.shares -= quantity
        portfolio_value = self.shares * price + self.cash
        mdd = self.get_current_mdd()
        self.mdd_history.append({'Date': date, 'MDD': mdd})
        self.portfolio.append({
            'Date': date,
            'Price': round(price, 1),
            'Action': action,
            'Amount': round(-amount, 1),
            'Shares Bought': round(-quantity, 1),
            '신규진입': False,
            '매수중지': False,
            '청산': True,
            f"기준 주가({self.signal_ticker})": round(self.signal_df['Close'].loc[date], 1),
            'Total MDD (%)': round(mdd, 1)
        })

    def get_current_mdd(self):
        if not self.daily_value:
            return 0
        values = [v['Value'] for v in self.daily_value]
        peak = max(values)
        current = values[-1]
        if peak == 0:
            return 0
        return (peak - current) / peak * 100

if __name__ == '__main__':
    st.markdown("## 📊 적립식 레버리지 매수 전략 시뮬레이터")

    ticker = st.text_input("매수 대상 티커 (예: TQQQ)", "TQQQ")
    signal_ticker = st.text_input("진입 조건 기준 티커 (예: QQQ)", "QQQ")
    start_date = st.date_input("시작일", pd.to_datetime("2021-01-01"))
    end_date = st.date_input("종료일", pd.to_datetime("2024-12-31"))
    initial_cash = st.number_input("최초 투자금 (달러)", value=100000, step=100, format="%d")
    entry_drawdown = st.number_input("고점 대비 하락률 (진입 조건) (%)", min_value=0, max_value=100, value=20, step=1)
    stop_buy_rally = st.number_input("고점 대비 상승률 (진입 중단 조건) (%)", min_value=0, max_value=100, value=5, step=1)
    exit_recovery = st.number_input("고점 대비 상승률 (청산 조건) (%)", min_value=0, max_value=100, value=25, step=1)
    per_buy_amount = st.number_input("1회 매수 금액 (달러)", value=2000, step=100, format="%d")
    buy_interval = st.number_input("정기 매수 간격 (일)", min_value=1, max_value=30, value=5, step=1)
    rebalance_interval = st.number_input("리밸런싱 간격 (일)", min_value=1, max_value=120, value=30, step=1)
    rebalance_target_stock_ratio = st.slider("리밸런싱 목표 주식 비중 (%)", min_value=0, max_value=100, value=80, step=5)

    chart_start = st.date_input("차트 보기 시작일", pd.to_datetime("2021-01-01"))
    chart_end = st.date_input("차트 보기 종료일", pd.to_datetime("2024-12-31"))

    if stop_buy_rally >= exit_recovery:
        st.error("❗ 청산 기준은 매수 중단 기준보다 커야 합니다. 값을 다시 설정해주세요.")
    
    elif st.button("시뮬레이션 실행"):
        sim = TQQQSimulator(
            ticker=ticker,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            per_buy_amount=per_buy_amount,
            buy_interval=buy_interval,
            initial_cash=initial_cash,
            entry_drawdown=entry_drawdown,
            exit_recovery=exit_recovery,
            stop_buy_rally=stop_buy_rally,
            signal_ticker=signal_ticker
        )
        result = sim.simulate(rebalance_interval=rebalance_interval, rebalance_target_stock_ratio=rebalance_target_stock_ratio)

        st.subheader("📌 시뮬레이션 결과")
        st.write(f"총 매수 수량: {result['총 매수 수량']:,} 주")
        st.write(f"최초 보유자산: {result['최초 보유자산']:,} 달러")
        st.write(f"최종 보유 주식 수: {result['보유 주식 수']:.0f} 주")
        st.write(f"최종 평가금액: {result['최종 평가금액']:,.0f} 달러")
        st.write(f"수익률: {result['수익률(%)']:.2f}%")
        st.write(f"Total MDD (%): {result['MDD(%)']:.2f}%")
        st.write(f"총 매수 횟수: {result['총 매수 횟수']} 회")

        st.subheader("📋 매수 기록")
        buy_df_display = result['매수 기록'].copy()
        ratio_df = pd.DataFrame(result['현금 비중'])
        buy_df_display = pd.merge(buy_df_display, ratio_df, on='Date', how='left')
        st.dataframe(buy_df_display, use_container_width=True)
        csv = buy_df_display.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 매수 기록 CSV 다운로드",
            data=csv,
            file_name='매수기록.csv',
            mime='text/csv'
        )

        st.subheader("📈 자산 추이 그래프")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(result['자산 추이']['Date'], result['자산 추이']['Value'], label="평가금액")
        ax.set_xlabel("날짜")
        ax.set_ylabel("평가금액")
        ax.set_title("자산 추이")
        ax.legend()
        st.pyplot(fig)

        st.subheader("📉 누적 주식 수 그래프")
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(result['누적 주식 수']['Date'], result['누적 주식 수']['Shares'], label="누적 주식 수", color='green')
        ax2.set_xlabel("날짜")
        ax2.set_ylabel("주식 수")
        ax2.set_title("누적 주식 수 추이")
        ax2.legend()
        st.pyplot(fig2)

        st.subheader("💵 현금 잔액 추이 그래프")
        cash_df = pd.DataFrame(result['현금 추이'])
        fig_cash, ax_cash = plt.subplots(figsize=(10, 4))
        ax_cash.plot(cash_df['Date'], cash_df['Cash'], label="현금 잔액", color='orange')
        ax_cash.set_xlabel("날짜")
        ax_cash.set_ylabel("현금")
        ax_cash.set_title("현금 잔액 추이")
        ax_cash.legend()
        st.pyplot(fig_cash)

        st.subheader("📊 현금 비중 (%) 추이")
        ratio_df = pd.DataFrame(result['현금 비중'])
        fig_ratio, ax_ratio = plt.subplots(figsize=(10, 3))
        ax_ratio.plot(ratio_df['Date'], ratio_df['CashRatio'] * 100, label="현금 비중 (%)", color='purple')
        ax_ratio.set_xlabel("날짜")
        ax_ratio.set_ylabel("%")
        ax_ratio.set_title("현금 비중 추이")
        ax_ratio.legend()
        st.pyplot(fig_ratio)

        if result['현금 부족 시점']:
            st.warning(f"⚠️ 현금 부족 발생 시점 {len(result['현금 부족 시점'])}건: 예: {result['현금 부족 시점'][0].strftime('%Y-%m-%d')}")

        st.subheader("📉 입력 기간 차트 보기")
        chart_df = sim.df[(sim.df.index >= pd.to_datetime(chart_start)) & (sim.df.index <= pd.to_datetime(chart_end))]
        if not chart_df.empty:
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            ax3.plot(chart_df.index, chart_df['Close'], label="종가")

            # 매수 시점 표시
            buy_df = result['매수 기록'].reset_index(drop=True)
            buy_df_in_range = buy_df[(buy_df['Date'] >= pd.to_datetime(chart_start)) & (buy_df['Date'] <= pd.to_datetime(chart_end))]
            buy_df_in_range = buy_df_in_range.fillna({'신규진입': False, '청산': False, '매수중지': False})
            ax3.scatter(buy_df_in_range[buy_df_in_range['신규진입']]['Date'], buy_df_in_range[buy_df_in_range['신규진입']]['Price'], color='blue', marker='^', label='신규진입')
            ax3.scatter(buy_df_in_range[buy_df_in_range['청산']]['Date'], buy_df_in_range[buy_df_in_range['청산']]['Price'], color='black', marker='v', label='청산')
            ax3.scatter(buy_df_in_range[buy_df_in_range['매수중지']]['Date'], buy_df_in_range[buy_df_in_range['매수중지']]['Price'], color='gray', marker='x', label='매수중지')
            ax3.set_title(f"{ticker} 차트 ({chart_start} ~ {chart_end})")
            ax3.set_xlabel("날짜")
            ax3.set_ylabel("가격")
            ax3.legend()
            st.pyplot(fig3)
        else:
            st.warning("선택한 기간에 데이터가 없습니다.")

