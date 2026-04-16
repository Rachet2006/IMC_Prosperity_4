import json
from typing import Any

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [self.compress_state(state, ""), self.compress_orders(orders), conversions, "", ""]
            )
        )
        max_item_length = (self.max_log_length - base_length) // 3
        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        return [[l.symbol, l.product, l.denomination] for l in listings.values()]

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        return {s: [od.buy_orders, od.sell_orders] for s, od in order_depths.items()}

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        return [
            [t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
            for arr in trades.values()
            for t in arr
        ]

    def compress_observations(self, observations: Observation) -> list[Any]:
        conv = {}
        for p, o in observations.conversionObservations.items():
            conv[p] = [o.bidPrice, o.askPrice, o.transportFees, o.exportTariff,
                       o.importTariff, o.sugarPrice, o.sunlightIndex]
        return [observations.plainValueObservations, conv]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        return [[o.symbol, o.price, o.quantity] for arr in orders.values() for o in arr]

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        lo, hi = 0, min(len(value), max_length)
        out = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."
            if len(json.dumps(candidate)) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1
        return out


logger = Logger()


# ---------------------------------------------------------------------------
# Constants — derived from 3-day data analysis
# ---------------------------------------------------------------------------

# ASH_COATED_OSMIUM
# - Fair value: fixed at 10,000 (mean=10000.20, std=5.35 across all days)
# - Dominant spread: 16 ticks (best_bid ≈ 9993-9994, best_ask ≈ 10009-10010)
# - Wall bids (vol≥25) cluster at 9991-9992; wall asks at 10012-10013
# - Lag-1 autocorr of mid changes: -0.50 (strong mean reversion, same as EMERALDS)
# - Trades spread across 9979-10026 (not just wall prices like EMERALDS)
ACO_SYM   = "ASH_COATED_OSMIUM"
ACO_LIMIT = 50
ACO_FAIR  = 10_000

# INTARIAN_PEPPER_ROOT
# - Perfect linear uptrend: +1,000 units/day, slope = 0.001002 per timestamp unit
# - Day starts: Day-2≈9998.5, Day-1≈10998.5, Day0≈11998.5 (+1000 each day)
# - Residual noise around trend: std≈2, max≈10 (almost perfectly linear)
# - Lag-1 autocorr of mid changes: -0.50 (mean reversion on top of trend)
# - Spread: 13-14 ticks (half-integers, e.g. 11998.5)
# - ~332 trades/day, avg qty 5.2, avg gap 3034 ticks between trades
PEPPER_SYM   = "INTARIAN_PEPPER_ROOT"
PEPPER_LIMIT = 50
PEPPER_SLOPE = 0.001002  # seashells per timestamp unit (verified across all 3 days)

POS_LIMITS = {
    ACO_SYM:    ACO_LIMIT,
    PEPPER_SYM: PEPPER_LIMIT,
}


# ---------------------------------------------------------------------------
# Persistent state (survives between run() calls via traderData)
# ---------------------------------------------------------------------------
class State:
    def __init__(self) -> None:
        # PEPPER: estimated price at timestamp=0 this day (updated on first tick)
        self.pepper_base: float | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_mid(order_depth: OrderDepth) -> float | None:
    """Best-bid/ask midpoint. Returns None if either side is empty."""
    if not order_depth.buy_orders or not order_depth.sell_orders:
        return None
    return (max(order_depth.buy_orders) + min(order_depth.sell_orders)) / 2

def get_best_bid(order_depth: OrderDepth) -> int | None:
    return max(order_depth.buy_orders) if order_depth.buy_orders else None

def get_best_ask(order_depth: OrderDepth) -> int | None:
    return min(order_depth.sell_orders) if order_depth.sell_orders else None


# ---------------------------------------------------------------------------
# ProductTrader base class
# ---------------------------------------------------------------------------
class ProductTrader:

    def __init__(self, name, state, prints, new_trader_data, product_group=None):

        self.orders = []

        self.name = name
        self.state = state
        self.prints = prints
        self.new_trader_data = new_trader_data
        self.product_group = name if product_group is None else product_group

        self.last_traderData = self.get_last_traderData()

        self.position_limit = POS_LIMITS.get(self.name, 0)
        self.initial_position = self.state.position.get(self.name, 0)
        self.expected_position = self.initial_position

        self.mkt_buy_orders, self.mkt_sell_orders = self.get_order_depth()
        self.bid_wall, self.wall_mid, self.ask_wall = self.get_walls()
        self.best_bid, self.best_ask = self.get_best_bid_ask()

        self.max_allowed_buy_volume, self.max_allowed_sell_volume = self.get_max_allowed_volume()
        self.total_mkt_buy_volume, self.total_mkt_sell_volume = self.get_total_market_buy_sell_volume()

    def get_last_traderData(self):
        last_traderData = {}
        try:
            if self.state.traderData != '':
                last_traderData = json.loads(self.state.traderData)
        except:
            pass
        return last_traderData

    def get_best_bid_ask(self):
        best_bid = best_ask = None
        try:
            if len(self.mkt_buy_orders) > 0:
                best_bid = max(self.mkt_buy_orders.keys())
            if len(self.mkt_sell_orders) > 0:
                best_ask = min(self.mkt_sell_orders.keys())
        except:
            pass
        return best_bid, best_ask

    def get_walls(self):
        bid_wall = wall_mid = ask_wall = None
        try: bid_wall = sorted(self.mkt_buy_orders.items(), key=lambda x: x[1], reverse=True)[0][0]
        except: pass
        try: ask_wall = sorted(self.mkt_sell_orders.items(), key=lambda x: x[1], reverse=True)[0][0]
        except: pass
        try: wall_mid = (bid_wall + ask_wall) / 2
        except: pass
        return bid_wall, wall_mid, ask_wall

    def get_total_market_buy_sell_volume(self):
        market_bid_volume = market_ask_volume = 0
        try:
            market_bid_volume = sum([v for p, v in self.mkt_buy_orders.items()])
            market_ask_volume = sum([v for p, v in self.mkt_sell_orders.items()])
        except:
            pass
        return market_bid_volume, market_ask_volume

    def get_max_allowed_volume(self):
        max_allowed_buy_volume = self.position_limit - self.initial_position
        max_allowed_sell_volume = self.position_limit + self.initial_position
        return max_allowed_buy_volume, max_allowed_sell_volume

    def get_order_depth(self):
        order_depth, buy_orders, sell_orders = {}, {}, {}
        try: order_depth: OrderDepth = self.state.order_depths[self.name]
        except: pass
        try: buy_orders = {bp: abs(bv) for bp, bv in sorted(order_depth.buy_orders.items(), key=lambda x: x[0], reverse=True)}
        except: pass
        try: sell_orders = {sp: abs(sv) for sp, sv in sorted(order_depth.sell_orders.items(), key=lambda x: x[0])}
        except: pass
        return buy_orders, sell_orders

    def bid(self, price, volume, logging=True):
        abs_volume = min(abs(int(volume)), self.max_allowed_buy_volume)
        order = Order(self.name, int(price), abs_volume)
        if logging: self.log("BUYO", {"p": price, "s": self.name, "v": int(volume)}, product_group='ORDERS')
        self.max_allowed_buy_volume -= abs_volume
        self.orders.append(order)

    def ask(self, price, volume, logging=True):
        abs_volume = min(abs(int(volume)), self.max_allowed_sell_volume)
        order = Order(self.name, int(price), -abs_volume)
        if logging: self.log("SELLO", {"p": price, "s": self.name, "v": int(volume)}, product_group='ORDERS')
        self.max_allowed_sell_volume -= abs_volume
        self.orders.append(order)

    def log(self, kind, message, product_group=None):
        if product_group is None: product_group = self.product_group
        if product_group == 'ORDERS':
            group = self.prints.get(product_group, [])
            group.append({kind: message})
        else:
            group = self.prints.get(product_group, {})
            group[kind] = message
        self.prints[product_group] = group

    def get_orders(self):
        return {}


# ---------------------------------------------------------------------------
# ASH_COATED_OSMIUM — AcoTrader
#
# Strategy: market-making anchored to ACO_FAIR = 10000.
# Typical book: bids 9991–9994, asks 10009–10013 (16-tick spread).
# wall_mid (outermost midpoint) ≈ 10002 — too high to use as anchor; use
# the fixed ACO_FAIR instead so taking/making logic is centred on 10000.
#
# Phase 1 — TAKING: immediately cross any ask < 10000 (underpriced) or any
#   bid > 10000 (overpriced). Also unwind inventory at fair value when off-centre.
#
# Phase 2 — MAKING: post the remaining capacity just inside the best existing
#   bid/ask, trying to earn the spread around fair value.
# ---------------------------------------------------------------------------
class AcoTrader(ProductTrader):
    def __init__(self, state, prints, new_trader_data):
        super().__init__(ACO_SYM, state, prints, new_trader_data)

    def get_orders(self):

        if not self.mkt_buy_orders and not self.mkt_sell_orders:
            return {self.name: self.orders}

        fair = ACO_FAIR
        microprice = (self.best_bid * self.total_mkt_sell_volume + self.best_ask * self.total_mkt_buy_volume) / (self.total_mkt_sell_volume +  self.total_mkt_buy_volume)

        ##########################################################
        ####### 1. TAKING
        ##########################################################
        for sp, sv in self.mkt_sell_orders.items():
            if sp <= self.wall_mid-1:
                self.bid(sp, sv, logging=False)
            elif sp <=self.wall_mid and self.initial_position < 0:
                volume = min(sv, abs(self.initial_position))
                self.bid(sp, volume, logging=False)

        for bp, bv in self.mkt_buy_orders.items():
            if bp >= self.wall_mid+1:
                self.ask(bp, bv, logging=False)
            elif bp >= self.wall_mid and self.initial_position > 0:
                volume = min(bv, self.initial_position)
                self.ask(bp, volume, logging=False)

        ###########################################################
        ####### 2. MAKING
        ###########################################################
        # Auction dynamics insight: post AT the wall price levels, not inside them.
        # The walls are the max-volume levels that anchor the auction clearing price.
        # Posting inside the walls puts us at a thinner level that may not clear.
        bid_price = int(self.bid_wall) if self.bid_wall is not None else fair - 8
        ask_price = int(self.ask_wall) if self.ask_wall is not None else fair + 8

        # Still try to overbid the best bid if it's between wall and fair
        for bp, bv in self.mkt_buy_orders.items():
            overbidding_price = bp + 1
            if bv > 1 and overbidding_price < self.wall_mid:
                bid_price = max(bid_price, overbidding_price)
            else:
                bid_price = max(bid_price, bp)
            break

        # Still try to underask the best ask if it's between wall and fair
        for sp, sv in self.mkt_sell_orders.items():
            underbidding_price = sp - 1
            if sv > 1 and underbidding_price > self.wall_mid:
                ask_price = min(ask_price, underbidding_price)
            else:
                ask_price = min(ask_price, sp)
            break

        # POST ORDERS
        self.bid(bid_price, self.max_allowed_buy_volume)
        self.ask(ask_price, self.max_allowed_sell_volume)

        return {self.name: self.orders}


# ---------------------------------------------------------------------------
# INTARIAN_PEPPER_ROOT — uptrend carry trade
#
# Key insight: price rises at exactly 0.001002 per timestamp unit (~10/iteration).
# This is NOT a mean-reversion product — it's a pure uptrend carry trade.
# Holding max long from the start of each day captures +1,000 units/day.
# The noise (std≈2) is negligible compared to the trend signal.
# Fair value at time t = pepper_base + PEPPER_SLOPE * t
#
# Auction dynamics insight (Rook-E1): the clearing price is where cumulative
# bid volume >= cumulative ask volume is maximised. For an uptrending product,
# buying at current ask prices IS profitable because the trend will cover the
# cost within a few ticks. Post passive bids at best_bid+1 to attract sellers
# and take all asks up to fair_value + 20 (trend buffer).
# Almost never sell — only post asks well above fair_value to avoid filling.
# ---------------------------------------------------------------------------
def pepper_orders(order_depth: OrderDepth, position: int, timestamp: int, state: State) -> list[Order]:
    orders: list[Order] = []

    mid = get_mid(order_depth)
    if mid is None:
        return orders

    buy_cap  = PEPPER_LIMIT - position
    sell_cap = PEPPER_LIMIT + position

    # Estimate base price (price at timestamp=0 this day) on first tick
    if state.pepper_base is None:
        state.pepper_base = mid - PEPPER_SLOPE * timestamp

    # Fair value at the current timestamp
    fair_value = state.pepper_base + PEPPER_SLOPE * timestamp

    # Phase 1: Aggressively take asks up to fair_value + 20
    # The uptrend means even paying slightly above fair value is profitable.
    for ap in sorted(order_depth.sell_orders.keys()):
        if ap > fair_value + 20 or buy_cap <= 0:
            break
        qty = min(abs(order_depth.sell_orders[ap]), buy_cap)
        orders.append(Order(PEPPER_SYM, ap, qty))
        buy_cap -= qty

    # Phase 2: Passive bid just above best existing bid to attract sellers
    # This positions us at the max-volume clearing level (auction dynamics).
    if buy_cap > 0 and order_depth.buy_orders:
        best_bid = max(order_depth.buy_orders.keys())
        passive_bid = int(best_bid) + 1
        # Only post if still below fair value (don't pay more than trend justifies)
        if passive_bid < fair_value + 5:
            orders.append(Order(PEPPER_SYM, passive_bid, buy_cap))

    # Phase 3: Post asks well above fair value — nearly impossible to fill,
    # just satisfies position limit rules if we somehow hit max long.
    if sell_cap > 0:
        ask_price = int(fair_value + 50)
        orders.append(Order(PEPPER_SYM, ask_price, -sell_cap))

    return orders


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------
class Trader:

    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}
        new_trader_data = {}
        prints = {
            "GENERAL": {
                "TIMESTAMP": state.timestamp,
                "POSITIONS": state.position,
            },
        }

        # Load persistent PEPPER state
        persistent = State()
        try:
            if state.traderData:
                data = json.loads(state.traderData)
                persistent.pepper_base = data.get("_pepper_base")
        except:
            pass

        # ASH_COATED_OSMIUM — StaticTrader market-making
        if ACO_SYM in state.order_depths:
            try:
                trader = AcoTrader(state, prints, new_trader_data)
                result.update(trader.get_orders())
            except:
                pass

        # INTARIAN_PEPPER_ROOT — uptrend carry strategy
        if PEPPER_SYM in state.order_depths:
            pos = state.position.get(PEPPER_SYM, 0)
            result[PEPPER_SYM] = pepper_orders(
                state.order_depths[PEPPER_SYM], pos, state.timestamp, persistent
            )

        # Persist state for next tick
        new_trader_data["_pepper_base"] = persistent.pepper_base
        try:
            final_trader_data = json.dumps(new_trader_data)
        except:
            final_trader_data = ''

        logger.flush(state, result, 0, final_trader_data)
        return result, 0, final_trader_data
