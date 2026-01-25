from edgar import set_identity, use_local_storage, Company, download_edgar_data
import pandas as pd
from typing import List
import os

class SecFilingsLoader:
    def __init__(self, user_agent: str):
        if not user_agent:
            raise ValueError("User Agent is required")

        user_agent = user_agent.strip()
        if " " not in user_agent:
            raise ValueError('User Agent must be "Name email@domain.com"')

        set_identity(user_agent)

        os.makedirs("edgar_cache", exist_ok=True)
        use_local_storage("edgar_cache", True)
        download_edgar_data(submissions=False, facts=False, reference=True)

    def load_company_filings(self, tickers: List[str]) -> pd.DataFrame:
        result = []

        totalFilings = 0
        totalRiskFactorsSections = 0
        totalManagementDiscussionSections = 0
        totalMarketRiskSections = 0
        
        for ticker in tickers:
            company = Company(ticker)
            filings = company.get_filings(
                year=range(2000, 2027),
                form=["10-K", "10-Q"]
            )
            
            totalFilings += len(filings)
            print(f"-------- Processing {ticker}, found {len(filings)} filings --------", flush=True)
            
            for filing in filings:
                print(f"Processing filing {filing.accession_no} ({filing.form}, {filing.filing_date})", flush=True)
                
                try:
                    filing_date = pd.to_datetime(filing.filing_date)
                    form_type = filing.form
                        
                    texts = self._extract_sections(filing.obj())
                    
                    if not texts:
                        continue

                    result.append({
                        "ticker": ticker,
                        "form_type": form_type,
                        "filing_date": filing_date,
                        "risk_factors_text": texts["risk_factors"],
                        "management_discussion_text": texts["management_discussion"],
                        "market_risk_text": texts["market_risk"],
                    })
                    
                    foundRiskFactors = texts["risk_factors"] != ""
                    foundManagementDiscussion = texts["management_discussion"] != ""
                    foundMarketRisk = texts["market_risk"] != ""
                    
                    totalRiskFactorsSections += 1 if foundRiskFactors else 0
                    totalManagementDiscussionSections += 1 if foundManagementDiscussion else 0
                    totalMarketRiskSections += 1 if foundMarketRisk else 0
                    
                    print(f"Risk Factors: {'✔️' if foundRiskFactors else '❌'}, Management Discussion: {'✔️' if foundManagementDiscussion else '❌'}, Market Risk: {'✔️' if foundMarketRisk else '❌'}", flush=True)

                except Exception as e:
                    print(f"Skipping filing {filing.accession_no}: {e}")
        
        print(f"-------- Summary --------")
        print(f"Total filings processed: {totalFilings}")
        print(f"Total Risk Factors sections found: {totalRiskFactorsSections}")
        print(f"Total Management Discussion sections found: {totalManagementDiscussionSections}")
        print(f"Total Market Risk sections found: {totalMarketRiskSections}")
        
        df = pd.DataFrame(result)
        
        return df.sort_values("filing_date").reset_index(drop=True)

    def _extract_sections(self, filing) -> str:        
        texts = {}
        
        sections = {
            "risk_factors": ["Risk Factors"],
            "management_discussion": ["Management’s Discussion", "Management's Discussion"],
            "market_risk": ["Market Risk"]
        }
            
        for section_name in sections.keys():
            texts[section_name] = ""
        
        for (section_name, section_titles) in sections.items():
            for section in filing.sections.values():
                section_text = section.text()
                
                title_matches = any([t.lower() in section_text[:100].lower() for t in section_titles])
                if title_matches:
                    texts[section_name] = section_text
                    
                    break
            
        return texts

import sys 

if len(sys.argv) < 2:
    print("Usage: python src/sec_filings_loader.py 'Name email@domain.com'")
    sys.exit(1)

user_agent = sys.argv[1]
print(f"Using User Agent: {user_agent}")

sec_filings_loader = SecFilingsLoader(
    user_agent=user_agent
)

filings = sec_filings_loader.load_company_filings(
    tickers=[
        "AAPL", "MSFT", "GOOG", "TSLA", "META", "NVDA", "AMZN", "AVGO", "WMT", "XOM",
        "MA", "AMD", "KO", "MCD", "BA", "TXN", "PFE", "DE", "CVS", "NKE",
    ]
)

filings.to_csv("sec_filings.csv", index=False)