import csv

class Stock:
    def __init__(self, name, prices):
        self.name = name
        self.prices = prices
    
    def to_csv(self, path, n):
        data = [["Date", "Open", "High", "Low", "Close", "Volume", "Moving Average", "Weighted Average",
        "Momentum", "Stochastic K", "Stochastic D", "Relative Strength", "Larry", "Accumulation", "CCI"]]
        for i in range(max(2 * n, 14), len(self.prices)):
            line = [self.prices[i].date, self.prices[i].opening, self.prices[i].high, self.prices[i].low, self.prices[i].closing, self.prices[i].volume,
            self.moving_average(n, i), self.weighted_average(14, i), self.momentum(n, i), self.stochastic_k(n, i),
            self.stochastic_d(n, i), self.relative_strength(n, i), self.larry(n, i),
            self.accumulation(i), self.CCI(n, i)]
            data.append(line)
        with open(path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in data:
                writer.writerow(row)
    
    def moving_average(self, n, t):
        n_prices = list(map(lambda p: p.closing, self.prices[t - n + 1: t + 1]))
        return sum(n_prices) / len(n_prices)
    
    def weighted_average(self, n, t):
        n_prices = list(map(lambda p: p.closing, self.prices[t - n + 1: t + 1]))
        weighted_average = 0
        for i in range(len(n_prices)):
            weighted_average += n_prices[i] * (i + 1)
        return weighted_average / sum(range(n + 1))
    
    def momentum(self, n, t):
        return self.prices[t].closing - self.prices[t - n + 1].closing

    def lowest_low(self, n, t):
        n_prices = list(map(lambda p: p.closing, self.prices[t - n + 1: t + 1]))
        return min(n_prices)
    
    def highest_high(self, n, t):
        n_prices = list(map(lambda p: p.closing, self.prices[t - n + 1: t + 1]))
        return max(n_prices)

    def stochastic_k(self, n, t):
        ll = self.lowest_low(n, t)
        hh = self.highest_high(n, t)
        try:
            return (self.prices[t].closing - ll) / (hh - ll) * 100
        except:
            return 0
    
    def stochastic_d(self, n, t):
        total = 0
        for i in range(n):
            total += self.stochastic_k(n, t - i)
        return total / n
    
    def up(self, t):
        cur = self.prices[t].closing
        prev = self.prices[t - 1].closing
        if cur - prev > 0:
            return (cur - prev) / prev
        else:
            return None
    
    def down(self, t):
        cur = self.prices[t].closing
        prev = self.prices[t - 1].closing
        if cur - prev < 0:
            return (prev - cur) / prev
        else:
            return None

    def relative_strength(self, n, t):
        up_total = 0
        up_days = 0
        down_total = 0
        down_days = 0
        for i in range(1, n):
            up = self.up(t - i)
            down = self.down(t - i)
            if up:
                up_total += up
                up_days += 1
            if down:
                down_total += down
                down_days += 1
        if up_days == 0:
            return 0
        if down_days == 0:
            return 100
        up_avg = (up_total / up_days)
        down_avg = (down_total / down_days)
        return 100 - (100 / (1 + (up_avg / down_avg)))

    def larry(self, n, t):
        ll = self.lowest_low(n, t)
        hh = self.highest_high(n, t)
        try:
            return (hh - self.prices[t].closing) / (hh - ll) * 100
        except:
            return 0

    def accumulation(self, t):
        cur = self.prices[t]
        try:
            return (cur.high - cur.closing) / (cur.high - cur.low)
        except:
            return 0
    
    def M(self, t):
        cur = self.prices[t]
        return (cur.high + cur.low + cur.closing) / 3

    def SM(self, n, t):
        total = 0
        for i in range(n):
            total += self.M(t - i)
        return total / n

    def D(self, n, t):
        total = 0
        for i in range(n):
            total += abs(self.M(t - i) - self.SM(n, t))
        return total / n

    def CCI(self, n, t):
        try:
            return (self.M(t) - self.SM(n, t)) / (0.015 * self.D(n, t))
        except:
            return 0

class Price:
    def __init__(self, date, opening, high, low, closing, volume):
        self.date = date
        self.opening = opening
        self.high = high
        self.low = low
        self.closing = closing
        self.volume = volume
