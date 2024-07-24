from datetime import datetime

def safe_date_cast(d)->datetime:
  if(d == None or not d):
    return datetime.now()
  return datetime.fromisoformat(d)