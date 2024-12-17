import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import telegram
import logging
from typing import List, Dict, Optional, Union

from core.logging_system import LoggingMixin

class AlertChannel:
    """
    Clase base abstracta para canales de alerta
    """
    def send(self, message: str) -> bool:
        """
        Método para enviar alerta
        
        :param message: Mensaje de alerta
        :return: Booleano indicando éxito del envío
        """
        raise NotImplementedError("Método de envío debe ser implementado")

class EmailAlertChannel(AlertChannel):
    """
    Canal de alertas por correo electrónico
    """
    def __init__(
        self, 
        smtp_host: str, 
        smtp_port: int,
        sender_email: str, 
        sender_password: str,
        recipients: List[str]
    ):
        """
        Inicializa el canal de alertas por email
        
        :param smtp_host: Servidor SMTP
        :param smtp_port: Puerto SMTP
        :param sender_email: Email remitente
        :param sender_password: Contraseña del remitente
        :param recipients: Lista de destinatarios
        """
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.recipients = recipients
    
    def send(self, message: str) -> bool:
        """
        Envía alerta por correo electrónico
        
        :param message: Mensaje de alerta
        :return: Booleano indicando éxito del envío
        """
        try:
            # Crear mensaje de correo
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = ', '.join(self.recipients)
            msg['Subject'] = 'Trading System Alert'
            
            msg.attach(MIMEText(message, 'plain'))
            
            # Establecer conexión SMTP
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            return True
        except Exception as e:
            logging.error(f"Error enviando alerta por email: {e}")
            return False

class TelegramAlertChannel(AlertChannel):
    """
    Canal de alertas por Telegram
    """
    def __init__(
        self, 
        bot_token: str, 
        chat_ids: List[str]
    ):
        """
        Inicializa el canal de alertas de Telegram
        
        :param bot_token: Token del bot de Telegram
        :param chat_ids: IDs de chats donde enviar alertas
        """
        self.bot = telegram.Bot(token=bot_token)
        self.chat_ids = chat_ids
    
    def send(self, message: str) -> bool:
        """
        Envía alerta por Telegram
        
        :param message: Mensaje de alerta
        :return: Booleano indicando éxito del envío
        """
        try:
            for chat_id in self.chat_ids:
                self.bot.send_message(
                    chat_id=chat_id, 
                    text=message, 
                    parse_mode=telegram.ParseMode.MARKDOWN
                )
            return True
        except Exception as e:
            logging.error(f"Error enviando alerta por Telegram: {e}")
            return False

class SlackAlertChannel(AlertChannel):
    """
    Canal de alertas por Slack
    """
    def __init__(
        self, 
        webhook_url: str
    ):
        """
        Inicializa el canal de alertas de Slack
        
        :param webhook_url: URL del webhook de Slack
        """
        self.webhook_url = webhook_url
    
    def send(self, message: str) -> bool:
        """
        Envía alerta por Slack
        
        :param message: Mensaje de alerta
        :return: Booleano indicando éxito del envío
        """
        try:
            response = requests.post(
                self.webhook_url, 
                json={'text': message}
            )
            return response.status_code == 200
        except Exception as e:
            logging.error(f"Error enviando alerta por Slack: {e}")
            return False

class AlertSystem(LoggingMixin):
    """
    Sistema centralizado de gestión de alertas
    """
    def __init__(
        self, 
        channels: Optional[List[AlertChannel]] = None
    ):
        """
        Inicializa el sistema de alertas
        
        :param channels: Canales de alerta predeterminados
        """
        self.channels = channels or []
    
    def add_channel(self, channel: AlertChannel):
        """
        Añade un canal de alerta
        
        :param channel: Canal de alerta a añadir
        """
        self.channels.append(channel)
    
    def remove_channel(self, channel_type: type):
        """
        Elimina canales de un tipo específico
        
        :param channel_type: Tipo de canal a eliminar
        """
        self.channels = [
            channel for channel in self.channels 
            if not isinstance(channel, channel_type)
        ]
    
    async def send_alert(
        self, 
        title: str, 
        message: str, 
        severity: str = 'info'
    ):
        """
        Envía una alerta a través de todos los canales
        
        :param title: Título de la alerta
        :param message: Mensaje de alerta
        :param severity: Severidad de la alerta
        """
        # Formatear mensaje
        formatted_message = f"*{title}*\n\n{message}\nSeveridad: {severity}"
        
        # Log de alerta
        log_method = {
            'info': self.logger.info,
            'warning': self.logger.warning,
            'error': self.logger.error,
            'critical': self.logger.critical
        }.get(severity.lower(), self.logger.info)
        
        log_method(formatted_message)
        
        # Enviar por todos los canales
        for channel in self.channels:
            try:
                channel.send(formatted_message)
            except Exception as e:
                self.logger.error(f"Error enviando alerta por canal: {e}")
    
    def filter_alerts(
        self, 
        alert_log: List[Dict], 
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Filtra alertas basado en criterios específicos
        
        :param alert_log: Log de alertas
        :param filters: Filtros a aplicar
        :return: Lista de alertas filtradas
        """
        if not filters:
            return alert_log
        
        return [
            alert for alert in alert_log
            if all(
                alert.get(key) == value 
                for key, value in filters.items()
            )
        ]

# Ejemplo de uso
def main():
    # Configurar sistema de alertas
    alert_system = AlertSystem()
    
    # Añadir canales de alerta
    alert_system.add_channel(
        EmailAlertChannel(
            smtp_host='smtp.gmail.com',
            smtp_port=587,
            sender_email='tu_email@gmail.com',
            sender_password='tu_contraseña',
            recipients=['destinatario@ejemplo.com']
        )
    )
    
    alert_system.add_channel(
        TelegramAlertChannel(
            bot_token='tu_token_de_bot',
            chat_ids=['tu_chat_id']
        )
    )
    
    alert_system.add_channel(
        SlackAlertChannel(
            webhook_url='tu_webhook_de_slack'
        )
    )
    
    # Enviar alerta de ejemplo
    async def example():
        await alert_system.send_alert(
            "Sistema de Trading", 
            "Operación ejecutada con éxito", 
            severity='info'
        )
    
    import asyncio
    asyncio.run(example())

if __name__ == "__main__":
    main()
