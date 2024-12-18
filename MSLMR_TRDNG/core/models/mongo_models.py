# core/models/mongo_models.py
from pymongo import MongoClient
from core.config_manager import ConfigManager

class MongoModels:
    def __init__(self):
        config = ConfigManager().get_config()
        self.client = MongoClient(
            host=config['MONGO_HOST'],
            port=config['MONGO_PORT'],
            username=config['MONGO_USER'],
            password=config['MONGO_PASSWORD']
        )
        self.db = self.client[config['MONGO_DB']]

    def get_collection(self, collection_name: str):
        """
        Get a specific collection from the database.
        :param collection_name: The name of the collection to get.
        :return: A MongoDB collection.
        """
        return self.db[collection_name]

    def insert_one(self, collection_name: str, document: dict):
        """
        Insert a single document into a collection.
        :param collection_name: The name of the collection.
        :param document: The document to insert.
        """
        collection = self.get_collection(collection_name)
        return collection.insert_one(document)

    def insert_many(self, collection_name: str, documents: list):
        """
        Insert multiple documents into a collection.
        :param collection_name: The name of the collection.
        :param documents: The documents to insert.
        """
        collection = self.get_collection(collection_name)
        return collection.insert_many(documents)

    def find_one(self, collection_name: str, query: dict):
        """
        Find a single document in a collection.
        :param collection_name: The name of the collection.
        :param query: The query to use.
        :return: The first matching document.
        """
        collection = self.get_collection(collection_name)
        return collection.find_one(query)

    def find(self, collection_name: str, query: dict):
        """
        Find multiple documents in a collection.
        :param collection_name: The name of the collection.
        :param query: The query to use.
        :return: A list of matching documents.
        """
        collection = self.get_collection(collection_name)
        return list(collection.find(query))

    def update_one(self, collection_name: str, query: dict, update: dict):
        """
        Update a single document in a collection.
        :param collection_name: The name of the collection.
        :param query: The query to use to find the document.
        :param update: The update to apply.
        """
        collection = self.get_collection(collection_name)
        return collection.update_one(query, {'$set': update})

    def delete_one(self, collection_name: str, query: dict):
        """
        Delete a single document from a collection.
        :param collection_name: The name of the collection.
        :param query: The query to use to find the document.
        """
        collection = self.get_collection(collection_name)
        return collection.delete_one(query)

    def delete_many(self, collection_name: str, query: dict):
        """
        Delete multiple documents from a collection.
        :param collection_name: The name of the collection.
        :param query: The query to use to find the documents.
        """
        collection = self.get_collection(collection_name)
        return collection.delete_many(query)
