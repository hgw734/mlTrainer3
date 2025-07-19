"""
Institutional-grade stock universe for momentum scanning.
500 stocks: 400 US promising stocks, 50 European ADRs, 50 Global ADRs.
"""

# US Technology Leaders (80 stocks)
US_TECHNOLOGY = [
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX',
    'ADBE', 'CRM', 'ORCL', 'NOW', 'INTU', 'AMD', 'AVGO', 'QCOM', 'TXN',
    'CSCO', 'ACN', 'IBM', 'PLTR', 'UBER', 'LYFT', 'SQ',
    'ZM', 'DOCN', 'NET', 'CRWD', 'OKTA', 'DDOG', 'MDB',
    'TEAM', 'WDAY', 'VEEV', 'SPLK', 'TWLO', 'TTD', 'PINS', 'SNAP',
    'ABNB', 'DASH', 'COIN', 'HOOD', 'AFRM', 'UPST', 'SOFI', 'PANW', 'FTNT',
    'CYBR', 'TENB', 'SAIL', 'VRNS', 'RBLX', 'U', 'MRVL', 'INTC', 'MU',
    'LRCX', 'KLAC', 'AMAT', 'MPWR', 'SMCI', 'DELL', 'HPQ', 'HPE', 'NTAP',
    'WDC', 'STX', 'ANET', 'FFIV', 'JNPR', 'CTSH', 'EPAM', 'GLW'
]

# US Healthcare & Biotech (70 stocks) - Removed VEEV duplicate
US_HEALTHCARE = [
    'JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'LLY', 'BMY', 'AMGN',
    'GILD', 'CVS', 'ANTM', 'CI', 'HUM', 'MDT', 'ISRG', 'VRTX', 'REGN', 'ZTS',
    'ILMN', 'BIIB', 'MRNA', 'BNTX', 'NVAX', 'TDOC', 'DXCM', 'ALGN',
    'IDXX', 'IQV', 'A', 'DHR', 'BSX', 'EW', 'SYK', 'ZBH', 'BDX', 'BAX',
    'TECH', 'HOLX', 'VAR', 'PODD', 'TMDX', 'EXAS', 'PACB', 'TWST', 'BEAM', 'CRSP',
    'INCY', 'BMRN', 'SGEN', 'RARE', 'BLUE', 'FOLD', 'SRPT', 'IONS', 'ACAD', 'SAGE',
    'PTCT', 'RGEN', 'UTHR', 'HALO', 'ARWR', 'EDIT', 'NTLA', 'FATE', 'CDNA', 'AGIO'
]

# US Financials (50 stocks)
US_FINANCIALS = [
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SCHW', 'USB',
    'PNC', 'TFC', 'COF', 'BK', 'STT', 'RF', 'FITB', 'HBAN', 'KEY', 'CFG',
    'CMA', 'ZION', 'NTRS', 'FRC', 'WAL', 'CBOE', 'CME', 'ICE', 'NDAQ',
    'SPGI', 'MCO', 'BRK-A', 'BRK-B', 'AIG', 'TRV', 'PGR', 'ALL', 'AFL', 'MET', 'PRU',
    'SYF', 'DFS', 'ALLY', 'MA', 'V', 'FIS', 'FISV', 'ADP', 'PAYX'
]

# US Consumer & Retail (40 stocks)
US_CONSUMER = [
    'WMT', 'HD', 'MCD', 'COST', 'LOW', 'TGT', 'SBUX', 'NKE', 'LULU', 'TJX',
    'KO', 'PEP', 'PG', 'CL', 'KMB', 'WBA', 'DG', 'DLTR', 'BBY', 'GPS',
    'M', 'JWN', 'NCLH', 'CCL', 'RCL', 'MAR', 'HLT', 'MGM', 'WYNN', 'LVS',
    'DIS', 'CMCSA', 'T', 'VZ', 'TMUS', 'CHTR', 'DISH', 'SIRI', 'EA', 'ATVI'
]

# US Energy & Materials (45 stocks) - Removed GOLD duplicate
US_ENERGY_MATERIALS = [
    'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'OXY', 'PXD', 'KMI', 'WMB', 'EPD',
    'ET', 'MPC', 'VLO', 'PSX', 'HES', 'APA', 'DVN', 'FANG', 'OVV', 'MRO',
    'FCX', 'NEM', 'AEM', 'SCCO', 'AA', 'DOW', 'LYB', 'CF', 'MOS',
    'APD', 'LIN', 'ECL', 'SHW', 'PPG', 'VMC', 'MLM', 'NUE', 'STLD', 'X',
    'CLF', 'MT', 'WLK', 'IFF', 'FMC'
]

# US Industrials & Aerospace (35 stocks)
US_INDUSTRIALS = [
    'BA', 'CAT', 'HON', 'UPS', 'UNP', 'LMT', 'RTX', 'GE', 'MMM', 'FDX',
    'CSX', 'NSC', 'DAL', 'AAL', 'UAL', 'LUV', 'JBLU', 'ALK', 'SAVE', 'HA',
    'NOC', 'GD', 'LDOS', 'TXT', 'LHX', 'HWM', 'PH', 'EMR', 'ITW', 'ROK',
    'DOV', 'XYL', 'IEX', 'FLS', 'GNRC'
]

# US REITs & Utilities (40 stocks)
US_REITS_UTILITIES = [
    'AMT', 'PLD', 'CCI', 'EQIX', 'SPG', 'O', 'WELL', 'EXR', 'AVB', 'EQR',
    'VTR', 'PEAK', 'DLR', 'PSA', 'ARE', 'BXP', 'HST', 'REG', 'MAC', 'SLG',
    'NEE', 'SO', 'DUK', 'AEP', 'EXC', 'XEL', 'WEC', 'ES', 'ED', 'PEG',
    'SRE', 'D', 'PCG', 'EIX', 'ETR', 'FE', 'AES', 'CMS', 'DTE', 'NI'
]

# European ADRs with Defense Focus (60 stocks) - Removed duplicates SHOP, ASX, GOLD
EUROPEAN_ADRS = [
    # Defense & Aerospace (8+ stocks including R3NK)
    'RKLB',  # Rocket Lab (New Zealand but trades on NASDAQ)
    'R3NK',  # Rank Group (UK) - defense/tech
    'PHG',   # Philips (Netherlands) - defense tech
    'RDS-A', # Shell (Netherlands/UK)
    'RDS-B', # Shell B shares
    'BP',    # BP (UK)
    'TM',    # Toyota (Japan) - defense contracts
    'LYG',   # Lloyds Banking (UK) - defense financing
    'BAE',   # BAE Systems (UK) - major defense contractor
    
    # Major European Companies
    'ASML',  # ASML (Netherlands) - semiconductor
    'SAP',   # SAP (Germany)
    'NVO',   # Novo Nordisk (Denmark)
    'UL',    # Unilever (Netherlands/UK)
    'NESN',  # Nestlé (Switzerland)
    'RHHBY', # Roche (Switzerland)
    'NVS',   # Novartis (Switzerland)
    'AZN',   # AstraZeneca (UK)
    'GSK',   # GlaxoSmithKline (UK)
    'BCS',   # Barclays (UK)
    'DB',    # Deutsche Bank (Germany)
    'CS',    # Credit Suisse (Switzerland)
    'UBS',   # UBS (Switzerland)
    'ING',   # ING Group (Netherlands)
    'SAN',   # Santander (Spain)
    'BBVA',  # BBVA (Spain)
    'BNP',   # BNP Paribas (France)
    'VOD',   # Vodafone (UK)
    'TEF',   # Telefonica (Spain)
    'ORAN',  # Orange (France)
    'NOK',   # Nokia (Finland)
    'ERIC',  # Ericsson (Sweden)
    'VOLVY', # Volvo (Sweden)
    'ADDYY', # Adidas (Germany)
    'BMWYY', # BMW (Germany)
    'VLKAY', # Volkswagen (Germany)
    'SIEGY', # Siemens (Germany)
    'BASFY', # BASF (Germany)
    'BAYRY', # Bayer (Germany)
    'TTE',   # TotalEnergies (France)
    'EQNR',  # Equinor (Norway)
    'STLA',  # Stellantis (Netherlands/Italy)
    'CNI',   # Canadian National Railway
    'RY',    # Royal Bank of Canada
    'TD',    # Toronto Dominion Bank
    'ENB',   # Enbridge (Canada)
    'SU',    # Suncor Energy (Canada)
    'ABX',   # Barrick Gold (Canada)
    'WPM',   # Wheaton Precious Metals
    'SPOT',  # Spotify (Sweden)
    'ARCC',  # Ares Capital Corp
    'MT',    # ArcelorMittal
    'EU',    # Encavis AG
    'DAI',   # Daimler AG
    'SE',    # Sea Limited
    'GRAB',  # Grab Holdings
    'CPNG',  # Coupang
    'XPEV',  # XPeng
    'LI'     # Li Auto
]

# Global ADRs (87 stocks) - Removed ASX, CYBR duplicates and added 37 new stocks
GLOBAL_ADRS = [
    # Asian Tech & Growth
    'BABA',  # Alibaba (China)
    'JD',    # JD.com (China)
    'PDD',   # PDD Holdings (China)
    'BIDU',  # Baidu (China)
    'NTES',  # NetEase (China)
    'TME',   # Tencent Music (China)
    'BILI',  # Bilibili (China)
    'WB',    # Weibo (China)
    'TSM',   # Taiwan Semiconductor
    'UMC',   # United Microelectronics
    'SONY',  # Sony (Japan)
    'NTT',   # NTT (Japan)
    'MUFG',  # Mitsubishi UFJ (Japan)
    'SMFG',  # Sumitomo Mitsui (Japan)
    'MFG',   # Mizuho Financial (Japan)
    'KB',    # KB Financial (South Korea)
    'SHG',   # Shinhan Financial (South Korea)
    'LPL',   # LG Display (South Korea)
    'PKX',   # POSCO (South Korea)
    
    # Latin American Growth
    'VALE',  # Vale (Brazil)
    'ITUB',  # Itau Unibanco (Brazil)
    'BBD',   # Banco Bradesco (Brazil)
    'ABEV',  # Ambev (Brazil)
    'PBR',   # Petrobras (Brazil)
    'MELI',  # MercadoLibre (Argentina)
    'GGAL',  # Grupo Galicia (Argentina)
    'YPF',   # YPF (Argentina)
    'CX',    # CEMEX (Mexico)
    'TV',    # Grupo Televisa (Mexico)
    'AMX',   # America Movil (Mexico)
    'FMX',   # FEMSA (Mexico)
    'KOF',   # Coca-Cola FEMSA (Mexico)
    
    # Indian Growth Stories
    'INFY',  # Infosys (India)
    'WIT',   # Wipro (India)
    'HDB',   # HDFC Bank (India)
    'IBN',   # ICICI Bank (India)
    'TTM',   # Tata Motors (India)
    'RDY',   # Dr. Reddy's (India)
    'WNS',   # WNS Holdings (India)
    
    # African & Middle Eastern
    'NMR',   # Nomura (Japan)
    'TEVA',  # Teva Pharmaceutical (Israel)
    'CHKP',  # Check Point Software (Israel)
    'NICE',  # NICE Systems (Israel)
    'WIX',   # Wix.com (Israel)
    'MNDY',  # Monday.com (Israel)
    'S',     # SentinelOne (Israel)
    'ZIM',   # ZIM Integrated Shipping
    'GLBE',  # Global-E Online (Israel)
    
    # Additional Elite Global Stocks (37 new additions)
    'NIO',   # Nio (China)
    'YUMC',  # Yum China Holdings
    'EDU',   # New Oriental Education
    'TAL',   # TAL Education Group
    'VIPS',  # Vipshop Holdings
    'DIDI',  # Didi Global
    'TIGR',  # UP Fintech Holding
    'FUTU',  # Futu Holdings
    'GOTU',  # Gaotu Techedu
    'IQ',    # iQIYI
    'DOYU',  # DouYu International
    'HUYA',  # HUYA Inc
    'LX',    # LexinFintech Holdings
    'QTT',   # Qutoutiao
    'BZUN',  # Baozun Inc
    'ATHM',  # Autohome Inc
    'RERE',  # ATRenew Inc
    'FINV',  # FinVolution Group
    'JMIA',  # Jumia Technologies
    'GOLD',  # Barrick Gold Corp
    'SHOP',  # Shopify Inc
    'TRP',   # TC Energy Corporation
    'CP',    # Canadian Pacific Railway
    'BNS',   # Bank of Nova Scotia
    'BMO',   # Bank of Montreal
    'CM',    # Canadian Imperial Bank
    'MFC',   # Manulife Financial
    'SLF',   # Sun Life Financial
    'FFH',   # Fairfax Financial Holdings
    'WCN',   # Waste Connections
    'TIH',   # Tharisa plc
    'NTR',   # Nutrien Ltd
    'K',     # Kinross Gold Corporation
    'PAAS',  # Pan American Silver
    'AUY',   # Yamana Gold Inc
    'EGO',   # Eldorado Gold Corporation
    'AU',    # AngloGold Ashanti Limited
    'BEKE',  # KE Holdings Inc
    'DOCU',  # DocuSign Inc
    'ROKU',  # Roku Inc
    'ZS',    # Zscaler Inc
    'SNOW',  # Snowflake Inc
    'PYPL',  # PayPal Holdings Inc
    'SHOP',  # Shopify Inc
    'NVTA'   # Invitae Corporation
]

# Complete 500-Stock Institutional Universe
INSTITUTIONAL_UNIVERSE = list(set(
    US_TECHNOLOGY + US_HEALTHCARE + US_FINANCIALS + US_CONSUMER + 
    US_ENERGY_MATERIALS + US_INDUSTRIALS + US_REITS_UTILITIES + 
    EUROPEAN_ADRS + GLOBAL_ADRS
))

# ETF Universe for market regime analysis
ETF_UNIVERSE = [
    'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'VEA', 'VWO',
    'XLF', 'XLK', 'XLV', 'XLE', 'XLI', 'XLU', 'XLB', 'XLP',
    'GLD', 'SLV', 'TLT', 'HYG', 'LQD', 'VIX', 'UVXY', 'SQQQ'
]

# High-volume liquid stocks for momentum scanning
HIGH_VOLUME_UNIVERSE = [
    'AAPL', 'TSLA', 'NVDA', 'AMD', 'SPY', 'QQQ', 'SQQQ', 'TQQQ',
    'AMZN', 'MSFT', 'GOOGL', 'META', 'NFLX', 'F', 'BAC', 'NIO'
]

# Sector mapping for analysis
SECTOR_MAPPING = {
    'US_Technology': US_TECHNOLOGY,
    'US_Healthcare': US_HEALTHCARE, 
    'US_Financials': US_FINANCIALS,
    'US_Consumer': US_CONSUMER,
    'US_Energy_Materials': US_ENERGY_MATERIALS,
    'US_Industrials': US_INDUSTRIALS,
    'US_REITs_Utilities': US_REITS_UTILITIES,
    'European_ADRs': EUROPEAN_ADRS,
    'Global_ADRs': GLOBAL_ADRS
}

def get_sector_stocks(sector: str) -> list:
    """Get stocks for a specific sector"""
    return SECTOR_MAPPING.get(sector, [])

def get_all_sectors() -> list:
    """Get list of all available sectors"""
    return list(SECTOR_MAPPING.keys())

def get_universe_by_type(universe_type: str = "institutional") -> list:
    """
    Get stock universe by type
    
    Args:
        universe_type: Type of universe ('institutional', 'etf', 'high_volume')
        
    Returns:
        List of stock symbols
    """
    if universe_type == "institutional":
        return INSTITUTIONAL_UNIVERSE
    elif universe_type == "etf":
        return ETF_UNIVERSE
    elif universe_type == "high_volume":
        return HIGH_VOLUME_UNIVERSE
    else:
        return INSTITUTIONAL_UNIVERSE

def get_total_universe_size() -> int:
    """Get total number of stocks in institutional universe"""
    return len(INSTITUTIONAL_UNIVERSE)