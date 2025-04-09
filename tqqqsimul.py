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
    def __init__(self, ticker="TQQQ", start_date="2020-01-01", end_date="2024-12-31", per_buy_amount=1_000_000, buy_interval=5, initial_cash=0, signal_ticker="TQQQ", entry_drawdown=20, exit_recovery=10, ):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.df = self.download_data()
        self.per_buy_amount = per_buy_amount
        self.buy_interval = buy_interval
        self.cash = initial_cash
        self.shares = 0
        self.portfolio = []
        self.daily_value = []
        self.sell_points = []  # 매도 시점 기록
        self.cumulative_shares = []  # 누적 주식 수 기록
        self.signal_ticker = signal_ticker
        self.entry_drawdown = entry_drawdown
        self.exit_recovery = exit_recovery
        self.signal_df = self.download_data(ticker=signal_ticker)
        self.signal_max = self.signal_df['Close'].cummax()
        self.compute_indicators()

    def download_data(self, ticker=None):
        if ticker is None:
            ticker = self.ticker
        df = yf.download(ticker, start=self.start_date, end=self.end_date)
        df = df[['Close']].copy()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.dropna(inplace=True)

        # st.subheader("📌 다운로드된 원시 데이터 확인")
        # st.write(df.head())
        # st.write("컬럼 목록:", df.columns.tolist())
        # st.write("데이터 타입:", str(type(df)))

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

    def simulate(self):
        current_week = None
        last_week_rsi = None
        peak_value = 0
        max_drawdown = 0

        in_position = False  # 진입 여부 플래그
        for i in range(len(self.df)):
            row = self.df.iloc[i]
            date, price, rsi, dev, week = row.name, row['Close'], row['RSI'], row['Deviation'], row['Week']

            if week != current_week:
                current_week = week
                last_week_rsi = rsi
                # if rsi >= 70 and self.shares > 0:
                #     self.sell(date, price, 'RSI70이상_10%매도', self.shares * 0.10)
                #     self.sell_points.append((date, price))

            # 전략 1: 진입 조건 - signal_ticker가 고점대비 X% 하락 (stop_rally 기준 이상 상승 시 진입 금지)
            signal_price = self.signal_df['Close'].iloc[i]
            signal_peak = self.signal_max.iloc[i]
            drawdown = (signal_price - signal_peak) / signal_peak * 100
            if not in_position and drawdown <= -self.entry_drawdown and signal_price <= signal_peak * (1 + self.exit_recovery / 100):
                self.buy(date, price, f'진입(DD {drawdown:.2f}%)', self.per_buy_amount)
                in_position = True

            # 전략 2: 청산 조건 - signal_ticker가 고점대비 X% 회복
            if in_position and drawdown >= -self.exit_recovery:
                self.sell(date, price, f'청산(DD {drawdown:.2f}%)', self.shares)
                self.sell_points.append((date, price))
                in_position = False

            if i % self.buy_interval == 0:
                self.buy(date, price, '정기매수', self.per_buy_amount)

            if not np.isnan(rsi) and not np.isnan(dev):
                amount = 0
                if rsi < 30 and dev < -10:
                    amount = self.per_buy_amount
                if rsi < 25 and dev < -15:
                    amount = self.per_buy_amount * 2
                if rsi < 20 and dev < -20:
                    amount = self.per_buy_amount * 3
                if amount > 0:
                    self.buy(date, price, '추가매수', amount)

            portfolio_value = self.shares * price
            peak_value = max(peak_value, portfolio_value)
            if peak_value > 0:
                drawdown = (peak_value - portfolio_value) / peak_value
                max_drawdown = max(max_drawdown, drawdown)

            self.daily_value.append({"Date": date, "Value": portfolio_value})
            self.cumulative_shares.append({"Date": date, "Shares": self.shares})

        final_price = self.df['Close'].iloc[-1]
        final_value = self.shares * final_price
        total_invested = sum(x['Amount'] for x in self.portfolio if x['Action'].startswith('정기매수') or x['Action'].startswith('추가매수'))
        return {
            '총 매수 금액': total_invested,
            '보유 주식 수': self.shares,
            '최종 평가금액': final_value,
            '수익률(%)': (final_value / total_invested - 1) * 100 if total_invested > 0 else 0,
            'MDD(%)': max_drawdown * 100,
            '매수 기록': pd.DataFrame(self.portfolio),
            '자산 추이': pd.DataFrame(self.daily_value),
            '누적 주식 수': pd.DataFrame(self.cumulative_shares),
            '총 매수 횟수': len([x for x in self.portfolio if x['Action'].startswith('정기매수') or x['Action'].startswith('추가매수')]),
            '매도 시점': self.sell_points
        }

    def buy(self, date, price, action, amount):
        if amount > self.cash:
            return  # 현금 부족 시 매수 불가
        quantity = amount / price
        self.cash -= amount
        self.shares += quantity
        self.portfolio.append({
            'Date': date, 'Price': price,
            'Action': action, 'Amount': amount,
            'Shares Bought': quantity
        })

    def sell(self, date, price, action, quantity):
        if quantity > self.shares:
            quantity = self.shares
        amount = quantity * price
        self.shares -= quantity
        self.portfolio.append({
            'Date': date, 'Price': price,
            'Action': action, 'Amount': -amount,
            'Shares Bought': -quantity
        })

if __name__ == '__main__':
    st.markdown("## 📊 TQQQ 전략 시뮬레이터")

    ticker = st.text_input("티커 입력", "TQQQ")
    start_date = st.date_input("시작일", pd.to_datetime("2020-01-01"))
    end_date = st.date_input("종료일", pd.to_datetime("2024-12-31"))
    initial_cash = st.number_input("최초 투자금 (원)", value=0, step=10000)
    entry_drawdown = st.number_input("진입 기준: 고점대비 하락률 (%)", min_value=0, max_value=100, value=20, step=1)
    exit_recovery = st.number_input("청산 기준: 고점대비 회복률 (%)", min_value=0, max_value=100, value=10, step=1)
    per_buy_amount = st.number_input("1회 매수 금액 (원)", value=1000000, step=10000)
    buy_interval = st.number_input("정기 매수 간격 (일)", min_value=1, max_value=30, value=5, step=1)

    chart_start = st.date_input("차트 보기 시작일", pd.to_datetime("2023-01-01"))
    chart_end = st.date_input("차트 보기 종료일", pd.to_datetime("2023-12-31"))

    if exit_recovery <= entry_drawdown:
        st.error("❗ 청산 기준은 진입 중단 기준보다 커야 합니다. 값을 다시 설정해주세요.")
    elif st.button("시뮬레이션 실행"):
        sim = TQQQSimulator(
            ticker=ticker,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            per_buy_amount=per_buy_amount,
            buy_interval=buy_interval,
            initial_cash=initial_cash,
            entry_drawdown=entry_drawdown,
            exit_recovery=exit_recovery
        )
        result = sim.simulate()

        st.subheader("📌 시뮬레이션 결과")
        st.write(f"총 매수 금액: {result['총 매수 금액']:,} 원")
        st.write(f"보유 주식 수: {result['보유 주식 수']:.4f} 주")
        st.write(f"최종 평가금액: {result['최종 평가금액']:,} 원")
        st.write(f"수익률: {result['수익률(%)']:.2f}%")
        st.write(f"MDD (최대 낙폭): {result['MDD(%)']:.2f}%")
        st.write(f"총 매수 횟수: {result['총 매수 횟수']} 회")

        st.subheader("📋 매수 기록")
        st.dataframe(result['매수 기록'], use_container_width=True)

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

        st.subheader("📉 입력 기간 차트 보기")
        chart_df = sim.df[(sim.df.index >= pd.to_datetime(chart_start)) & (sim.df.index <= pd.to_datetime(chart_end))]
        if not chart_df.empty:
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            ax3.plot(chart_df.index, chart_df['Close'], label="종가")

            # 매수 시점 표시
            buy_df = result['매수 기록']
            buy_df_in_range = buy_df[(buy_df['Date'] >= pd.to_datetime(chart_start)) & (buy_df['Date'] <= pd.to_datetime(chart_end))]
            ax3.scatter(buy_df_in_range['Date'], buy_df_in_range['Price'], color='red', marker='^', label='매수 시점')
            ax3.set_title(f"{ticker} 차트 ({chart_start} ~ {chart_end})")
            ax3.set_xlabel("날짜")
            ax3.set_ylabel("가격")
            ax3.legend()
            st.pyplot(fig3)
        else:
            st.warning("선택한 기간에 데이터가 없습니다.")
