import re
import json
from typing import Any, Dict, List, Optional, Union
import jsonschema
import email_validator
from decimal import Decimal, InvalidOperation

class DataValidator:
    """
    Clase de validación para diferentes tipos de datos
    """
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """
        Valida una dirección de correo electrónico
        
        :param email: Dirección de correo a validar
        :return: Booleano indicando validez del email
        """
        try:
            email_validator.validate_email(email)
            return True
        except email_validator.EmailNotValidError:
            return False
    
    @staticmethod
    def validate_phone(phone: str) -> bool:
        """
        Valida un número de teléfono
        
        :param phone: Número de teléfono a validar
        :return: Booleano indicando validez del teléfono
        """
        # Expresión regular para validar números de teléfono internacionales
        phone_regex = r'^\+?1?\d{9,15}$'
        return bool(re.match(phone_regex, phone))
    
    @staticmethod
    def validate_json(json_str: str) -> bool:
        """
        Valida si un string es JSON válido
        
        :param json_str: String JSON a validar
        :return: Booleano indicando validez del JSON
        """
        try:
            json.loads(json_str)
            return True
        except (json.JSONDecodeError, TypeError):
            return False
    
    @staticmethod
    def validate_numeric(
        value: Union[str, int, float, Decimal], 
        min_value: Optional[float] = None, 
        max_value: Optional[float] = None
    ) -> bool:
        """
        Valida un valor numérico con límites opcionales
        
        :param value: Valor a validar
        :param min_value: Valor mínimo permitido
        :param max_value: Valor máximo permitido
        :return: Booleano indicando validez del número
        """
        try:
            # Convertir a Decimal para validación precisa
            num = Decimal(str(value))
            
            # Validar límites si se proporcionan
            if min_value is not None and num < min_value:
                return False
            
            if max_value is not None and num > max_value:
                return False
            
            return True
        except (InvalidOperation, TypeError):
            return False
    
    @staticmethod
    def validate_schema(
        data: Union[Dict, List], 
        schema: Dict[str, Any]
    ) -> bool:
        """
        Valida un diccionario o lista contra un esquema JSON
        
        :param data: Datos a validar
        :param schema: Esquema de validación
        :return: Booleano indicando validez de los datos
        """
        try:
            jsonschema.validate(instance=data, schema=schema)
            return True
        except jsonschema.exceptions.ValidationError:
            return False
    
    @staticmethod
    def validate_trade_data(trade_data: Dict[str, Any]) -> bool:
        """
        Esquema de validación específico para datos de trading
        
        :param trade_data: Datos de trade a validar
        :return: Booleano indicando validez de los datos
        """
        trade_schema = {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "side": {"enum": ["buy", "sell"]},
                "entry_price": {"type": "number", "minimum": 0},
                "exit_price": {"type": "number", "minimum": 0},
                "size": {"type": "number", "minimum": 0},
                "profit_loss": {"type": "number"}
            },
            "required": ["symbol", "side", "entry_price", "size"]
        }
        
        return DataValidator.validate_schema(trade_data, trade_schema)
    
    @staticmethod
    def sanitize_input(input_str: str) -> str:
        """
        Sanitiza un string de entrada eliminando caracteres potencialmente dañinos
        
        :param input_str: String a sanitizar
        :return: String sanitizado
        """
        # Eliminar caracteres especiales y potencialmente dañinos
        return re.sub(r'[<>"\']', '', input_str)
    
    @classmethod
    def validate_strategy_config(
        cls, 
        config: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """
        Valida una configuración de estrategia
        
        :param config: Configuración de estrategia a validar
        :return: Diccionario de errores de validación
        """
        errors = {}
        
        # Validaciones de configuración de estrategia
        if 'name' not in config:
            errors['name'] = ['Campo requerido']
        
        if 'risk_per_trade' in config:
            if not cls.validate_numeric(
                config['risk_per_trade'], 
                min_value=0, 
                max_value=0.1
            ):
                errors['risk_per_trade'] = ['Debe ser un número entre 0 y 0.1']
        
        # Validar parámetros específicos de estrategia
        if 'parameters' in config:
            for param, value in config['parameters'].items():
                # Ejemplo de validación específica
                if param == 'rsi_period' and not cls.validate_numeric(
                    value, 
                    min_value=7, 
                    max_value=21
                ):
                    errors.setdefault('parameters', []).append(
                        f'RSI period inválido: {value}'
                    )
        
        return errors

# Ejemplo de uso
def main():
    # Ejemplos de validación
    validator = DataValidator()
    
    # Validar email
    print("Email válido:", validator.validate_email("usuario@ejemplo.com"))
    
    # Validar teléfono
    print("Teléfono válido:", validator.validate_phone("+1234567890"))
    
    # Validar JSON
    json_str = '{"symbol": "BTC/USDT", "side": "buy", "price": 50000}'
    print("JSON válido:", validator.validate_json(json_str))
    
    # Validar datos de trade
    trade_data = {
        "symbol": "BTC/USDT",
        "side": "buy",
        "entry_price": 50000,
        "size": 0.1
    }
    print("Datos de trade válidos:", validator.validate_trade_data(trade_data))
    
    # Validar configuración de estrategia
    strategy_config = {
        "name": "RSI Strategy",
        "risk_per_trade": 0.02,
        "parameters": {
            "rsi_period": 14
        }
    }
    validation_errors = validator.validate_strategy_config(strategy_config)
    print("Errores de validación:", validation_errors)

if __name__ == "__main__":
    main()
