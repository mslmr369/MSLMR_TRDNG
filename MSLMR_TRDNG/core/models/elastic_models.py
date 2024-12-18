# core/models/elastic_models.py
from elasticsearch_dsl import Document, Date, Keyword, Float

class Prediction(Document):
    timestamp = Date()
    symbol = Keyword()
    prediction = Float()
    confidence = Float()
    model_version = Keyword()

    class Index:
        name = 'ml_predictions'

class MarketAnalysis(Document):
    timestamp = Date()
    symbol = Keyword()
    metrics = Float()

    class Index:
        name = 'market_analysis'

