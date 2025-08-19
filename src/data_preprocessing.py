import re

def clean_text(text: str) -> str:
    """
    Basic text cleaning: remove HTML tags, URLs, special chars
    """
    text = re.sub(r"<.*?>", " ", text)      # remove HTML
    text = re.sub(r"http\S+|www\S+", " ", text)  # remove URLs
    text = re.sub(r"[^a-zA-Z']", " ", text) # keep only letters
    text = text.lower().strip()
    return text

# text='''<h1>AMAZING Product!!!</h1> 😍😍  
# I bought this phone on 12/08/2024 for ₹15,999... and wow!!! It's sooo good.  
# Battery life = 2 days 🔋, Camera = 108MP 📷, Display is <b>awesome</b>.  
# BUT... delivery was late :( arrived after 5 days!!!  
# Also, seller said "ORIGINAL", but I think accessories are duplicate 🤔😡.  
# Overall: 4.5/5 ⭐⭐⭐⭐☆
# '''

# print(clean_text(text))