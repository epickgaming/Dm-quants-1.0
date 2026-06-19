//+------------------------------------------------------------------+
//|                                                     PropAlgo.mq5  |
//|   Regime-filtered mean-reversion EA, fed by signals.csv from the  |
//|   Python "brain". Enforces prop-firm risk limits on every order.  |
//|                                                                    |
//|   The EA NEVER invents trades: it only acts on fresh rows in       |
//|   signals.csv (de-duplicated by symbol+asof). All strategy logic   |
//|   (pattern discovery, regime gate, direction, brackets) lives in   |
//|   Python; the EA is the risk-aware execution layer.                |
//+------------------------------------------------------------------+
#property copyright "PropAlgo"
#property version   "1.00"
#property strict

#include <Trade/Trade.mqh>

//--- Inputs -------------------------------------------------------------------
input string  InpSignalsFile      = "signals.csv";   // signals file (MQL5/Files or common)
input bool    InpUseCommonFolder  = false;            // read from terminal common folder
input double  InpRiskPct          = 0.005;            // risk per trade (0.5%)
input double  InpDailyLossCutoff  = 0.02;             // stop new trades after -2% on the day
input double  InpMaxDrawdown      = 0.10;             // hard halt at -10% of start balance
input double  InpMaxConcurrentRisk= 0.03;             // cap total open risk (~3% equity)
input int     InpWeeklyTradeCap   = 6;                // max new trades per ISO week (basket)
input bool    InpLive             = false;            // false = demo/paper guard (must set true to trade live)
input long    InpMagic            = 990011;           // EA magic number
input int     InpSlippagePoints   = 20;               // max deviation (points)
input int     InpPollSeconds      = 60;               // poll cadence for the signals file
//--- Broker symbol map (canonical -> broker). Leave a slot blank to ignore it.
input string  InpMapXAUUSD        = "XAUUSD";
input string  InpMapFTSE100       = "UK100";
input string  InpMapSP500         = "US500";
input string  InpMapCOPPER        = "COPPER";
input string  InpMapDAX           = "GER40";

//--- Globals ------------------------------------------------------------------
CTrade        trade;
string        g_processed[];       // "SYMBOL|ASOF" already acted on
datetime      g_lastPoll       = 0;
double        g_startBalance   = 0.0;
double        g_peakEquity     = 0.0;
bool          g_halted         = false;
datetime      g_dayStart       = 0;
double        g_dayStartEquity = 0.0;
int           g_weekId         = -1;
int           g_weekTradeCount = 0;

//+------------------------------------------------------------------+
int OnInit()
{
   trade.SetExpertMagicNumber(InpMagic);
   trade.SetDeviationInPoints(InpSlippagePoints);
   trade.SetTypeFillingBySymbol(_Symbol);

   g_startBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   g_peakEquity   = AccountInfoDouble(ACCOUNT_EQUITY);
   ArrayResize(g_processed, 0);

   PrintFormat("[PropAlgo] init: start balance=%.2f  live=%s  signals=%s",
               g_startBalance, (InpLive ? "TRUE" : "FALSE(paper)"), InpSignalsFile);
   if(!InpLive)
      Print("[PropAlgo] PAPER MODE: orders are blocked. Set InpLive=true to enable real orders.");
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
void OnTick()
{
   // Poll on a cadence (and effectively once per new H1 bar).
   datetime now = TimeCurrent();
   if(now - g_lastPoll < InpPollSeconds)
      return;
   g_lastPoll = now;

   UpdateRiskState();
   if(g_halted)
      return;

   ProcessSignals();
}

//+------------------------------------------------------------------+
//| Maintain daily / drawdown / weekly risk state                    |
//+------------------------------------------------------------------+
void UpdateRiskState()
{
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   if(equity > g_peakEquity) g_peakEquity = equity;

   // Hard total-drawdown halt (measured from start balance).
   double floor = g_startBalance * (1.0 - InpMaxDrawdown);
   if(equity <= floor)
   {
      if(!g_halted)
         PrintFormat("[PropAlgo] HARD HALT: equity %.2f <= floor %.2f (-%.0f%%). No further trades.",
                     equity, floor, InpMaxDrawdown * 100.0);
      g_halted = true;
      return;
   }

   // New trading day -> snapshot day-start equity.
   MqlDateTime dt; TimeToStruct(TimeCurrent(), dt);
   datetime dayKey = StringToTime(StringFormat("%04d.%02d.%02d", dt.year, dt.mon, dt.day));
   if(dayKey != g_dayStart)
   {
      g_dayStart = dayKey;
      g_dayStartEquity = equity;
   }

   // New ISO week -> reset the weekly trade counter.
   int wid = IsoWeekId(TimeCurrent());
   if(wid != g_weekId)
   {
      g_weekId = wid;
      g_weekTradeCount = 0;
   }
}

//+------------------------------------------------------------------+
//| Read signals.csv and act on fresh, allowed rows                  |
//+------------------------------------------------------------------+
void ProcessSignals()
{
   int flags = FILE_READ | FILE_CSV | FILE_ANSI | FILE_SHARE_READ;
   if(InpUseCommonFolder) flags |= FILE_COMMON;

   int h = FileOpen(InpSignalsFile, flags, ',');
   if(h == INVALID_HANDLE)
   {
      // Not an error: the brain may not have written yet.
      return;
   }

   // Header: symbol,timeframe,direction,entry,stop,target,atr,risk_pct,er,asof
   string cols[];
   bool headerSkipped = false;
   while(!FileIsEnding(h))
   {
      // Read one logical CSV row (10 fields).
      string symbol    = FileReadString(h);
      if(!headerSkipped)
      {
         // consume the rest of the header line
         for(int k = 0; k < 9 && !FileIsLineEnding(h); k++) FileReadString(h);
         headerSkipped = true;
         continue;
      }
      string timeframe = FileReadString(h);
      string direction = FileReadString(h);
      double entry     = (double)FileReadString(h);
      double stop      = (double)FileReadString(h);
      double target    = (double)FileReadString(h);
      double atr       = (double)FileReadString(h);
      double riskpct   = (double)FileReadString(h);
      double er        = (double)FileReadString(h);
      string asof      = FileReadString(h);

      if(StringLen(symbol) == 0) continue;

      string brokerSym = MapSymbol(symbol);
      string key = brokerSym + "|" + asof;
      if(AlreadyProcessed(key)) continue;

      // Mark processed regardless of outcome so we never double-fire.
      AddProcessed(key);
      HandleSignal(brokerSym, direction, entry, stop, target, atr, riskpct, er, asof);
   }
   FileClose(h);
}

//+------------------------------------------------------------------+
//| Validate risk gates and place the order for one signal           |
//+------------------------------------------------------------------+
void HandleSignal(string sym, string dir, double entry, double stop, double target,
                  double atr, double riskpct, double er, string asof)
{
   if(g_halted) return;

   if(!SymbolSelect(sym, true))
   {
      PrintFormat("[PropAlgo] SKIP %s @%s: symbol not available at broker", sym, asof);
      return;
   }

   // One position per symbol.
   if(PositionSelect(sym))
   {
      PrintFormat("[PropAlgo] SKIP %s @%s: position already open", sym, asof);
      return;
   }

   // Daily -2% cutoff (based on realised+floating since day start).
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   if(g_dayStartEquity > 0 &&
      (equity - g_dayStartEquity) / g_dayStartEquity <= -InpDailyLossCutoff)
   {
      PrintFormat("[PropAlgo] SKIP %s @%s: daily loss cutoff hit (-%.1f%%)",
                  sym, asof, InpDailyLossCutoff * 100.0);
      return;
   }

   // Weekly cap.
   if(g_weekTradeCount >= InpWeeklyTradeCap)
   {
      PrintFormat("[PropAlgo] SKIP %s @%s: weekly cap %d reached", sym, asof, InpWeeklyTradeCap);
      return;
   }

   // Concurrent open-risk cap.
   double openRisk = CurrentOpenRiskFraction();
   double thisRisk = (riskpct > 0 ? riskpct : InpRiskPct);
   if(openRisk + thisRisk > InpMaxConcurrentRisk + 1e-9)
   {
      PrintFormat("[PropAlgo] SKIP %s @%s: concurrent risk cap (open=%.3f + new=%.3f > %.3f)",
                  sym, asof, openRisk, thisRisk, InpMaxConcurrentRisk);
      return;
   }

   // Sizing: lots = money_risk / (sl_distance_in_price / tickSize * tickValue).
   double slDistance = MathAbs(entry - stop);
   if(slDistance <= 0)
   {
      PrintFormat("[PropAlgo] SKIP %s @%s: invalid stop distance", sym, asof);
      return;
   }
   double lots = ComputeLots(sym, slDistance, thisRisk);
   if(lots <= 0)
   {
      PrintFormat("[PropAlgo] SKIP %s @%s: computed lots <= 0", sym, asof);
      return;
   }

   bool isBuy = (StringCompare(dir, "BUY", false) == 0);

   if(!InpLive)
   {
      PrintFormat("[PropAlgo] PAPER %s %s lots=%.2f entry~%.5f sl=%.5f tp=%.5f er=%.3f @%s (set InpLive=true to send)",
                  (isBuy ? "BUY" : "SELL"), sym, lots, entry, stop, target, er, asof);
      g_weekTradeCount++; // count paper intents too, so the cap behaves identically
      return;
   }

   bool ok;
   if(isBuy) ok = trade.Buy(lots, sym, 0.0, stop, target, "PropAlgo");
   else      ok = trade.Sell(lots, sym, 0.0, stop, target, "PropAlgo");

   if(ok)
   {
      g_weekTradeCount++;
      PrintFormat("[PropAlgo] ORDER %s %s lots=%.2f sl=%.5f tp=%.5f er=%.3f @%s -> ticket %I64u",
                  (isBuy ? "BUY" : "SELL"), sym, lots, stop, target, er, asof, trade.ResultOrder());
   }
   else
   {
      PrintFormat("[PropAlgo] FAIL %s %s lots=%.2f retcode=%d (%s)",
                  (isBuy ? "BUY" : "SELL"), sym, lots, trade.ResultRetcode(),
                  trade.ResultRetcodeDescription());
   }
}

//+------------------------------------------------------------------+
//| Position sizing                                                  |
//+------------------------------------------------------------------+
double ComputeLots(string sym, double slDistance, double riskFraction)
{
   double equity    = AccountInfoDouble(ACCOUNT_EQUITY);
   double moneyRisk = equity * riskFraction;

   double tickValue = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_VALUE);
   double tickSize  = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_SIZE);
   if(tickValue <= 0 || tickSize <= 0)
      return 0.0;

   // Loss per 1.0 lot if stopped out.
   double lossPerLot = (slDistance / tickSize) * tickValue;
   if(lossPerLot <= 0) return 0.0;

   double lots = moneyRisk / lossPerLot;

   // Normalise to the broker's volume step / min / max.
   double step = SymbolInfoDouble(sym, SYMBOL_VOLUME_STEP);
   double vmin = SymbolInfoDouble(sym, SYMBOL_VOLUME_MIN);
   double vmax = SymbolInfoDouble(sym, SYMBOL_VOLUME_MAX);
   if(step > 0) lots = MathFloor(lots / step) * step;
   if(lots < vmin) lots = 0.0;            // too small to take within risk -> skip
   if(vmax > 0 && lots > vmax) lots = vmax;
   return lots;
}

//+------------------------------------------------------------------+
//| Sum of open-position risk as a fraction of equity                |
//+------------------------------------------------------------------+
double CurrentOpenRiskFraction()
{
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   if(equity <= 0) return 0.0;
   double risk = 0.0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetInteger(POSITION_MAGIC) != InpMagic) continue;
      string sym  = PositionGetString(POSITION_SYMBOL);
      double open = PositionGetDouble(POSITION_PRICE_OPEN);
      double sl   = PositionGetDouble(POSITION_SL);
      double vol  = PositionGetDouble(POSITION_VOLUME);
      if(sl <= 0) continue;
      double tickValue = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_VALUE);
      double tickSize  = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_SIZE);
      if(tickValue <= 0 || tickSize <= 0) continue;
      double lossMoney = (MathAbs(open - sl) / tickSize) * tickValue * vol;
      risk += lossMoney / equity;
   }
   return risk;
}

//+------------------------------------------------------------------+
//| Helpers                                                          |
//+------------------------------------------------------------------+
string MapSymbol(string canonical)
{
   if(canonical == "XAUUSD"  && StringLen(InpMapXAUUSD)  > 0) return InpMapXAUUSD;
   if(canonical == "FTSE100" && StringLen(InpMapFTSE100) > 0) return InpMapFTSE100;
   if(canonical == "SP500"   && StringLen(InpMapSP500)   > 0) return InpMapSP500;
   if(canonical == "COPPER"  && StringLen(InpMapCOPPER)  > 0) return InpMapCOPPER;
   if(canonical == "DAX"     && StringLen(InpMapDAX)      > 0) return InpMapDAX;
   // Already a broker symbol (Python may have mapped it) -> use as-is.
   return canonical;
}

bool AlreadyProcessed(string key)
{
   for(int i = ArraySize(g_processed) - 1; i >= 0; i--)
      if(g_processed[i] == key) return true;
   return false;
}

void AddProcessed(string key)
{
   int n = ArraySize(g_processed);
   ArrayResize(g_processed, n + 1);
   g_processed[n] = key;
}

int IsoWeekId(datetime t)
{
   // Cheap, monotonic week id: floor(days since epoch / 7). Good enough to
   // reset a weekly counter; not a strict ISO-8601 week number.
   return (int)(t / (7 * 24 * 60 * 60));
}
//+------------------------------------------------------------------+
