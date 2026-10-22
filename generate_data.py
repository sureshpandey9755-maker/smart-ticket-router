import pandas as pd


data = {
    'text': [
        "My order hasn't arrived yet and it's been 5 days.",
        "The product I received is damaged and broken. I want a refund.",
        "How do I track my shipment? The link isn't working.",
        "I was charged twice for the same order, please fix this payment issue.",
        "The size of the shirt is too small, I need an exchange.",
        "Very poor service, I am extremely angry and disappointed!",
        "Thank you so much for resolving my issue quickly. Great support!",
        "Where is my invoice? I need it for tax purposes.",
        "Can I change my delivery address for order #12345?",
        "The payment failed but money was debited from my bank account."
    ] * 20, 
    'department': [
        "Shipping", "Returns", "Shipping", "Billing", 
        "Returns", "Support", "Support", "Billing", "Shipping", "Billing"
    ] * 20,
    'sentiment': [
        "Negative", "Negative", "Neutral", "Negative", 
        "Neutral", "Negative", "Positive", "Neutral", "Neutral", "Negative"
    ] * 20
}


df = pd.DataFrame(data)
df.to_csv('data/tickets.csv', index=False)

print("Dataset generated successfully and saved to data/tickets.csv!")