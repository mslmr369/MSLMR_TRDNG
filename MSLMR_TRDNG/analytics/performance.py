import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional

class PerformanceAnalyzer:
    """
    Herramienta avanzada para análisis de rendimiento de estrategias de trading
    """
    
    def __init__(self, trades_data: List[Dict]):
        """
        Inicializa el analizador de rendimiento
        
        :param trades_data: Lista de trades con detalles
        """
        self.trades_df = pd.DataFrame(trades_data)
        self.trades_df['timestamp'] = pd.to_datetime(self.trades_df['timestamp'])
    
    def calculate_returns(self) -> pd.Series:
        """
        Calcula retornos acumulados
        
        :return: Serie de retornos acumulados
        """
        cumulative_returns = (1 + self.trades_df['profit_loss'] / 100).cumprod() - 1
        return cumulative_returns
    
    def calculate_performance_metrics(self) -> Dict:
        """
        Calcula métricas detalladas de rendimiento
        
        :return: Diccionario de métricas
        """
        # Preparar datos
        total_trades = len(self.trades_df)
        winning_trades = self.trades_df[self.trades_df['profit_loss'] > 0]
        losing_trades = self.trades_df[self.trades_df['profit_loss'] < 0]
        
        metrics = {
            # Métricas básicas
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / total_trades * 100,
            
            # Rendimiento
            'total_profit': self.trades_df['profit_loss'].sum(),
            'average_profit': self.trades_df['profit_loss'].mean(),
            'median_profit': self.trades_df['profit_loss'].median(),
            
            # Análisis de riesgo
            'max_drawdown': self._calculate_max_drawdown(),
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'sortino_ratio': self._calculate_sortino_ratio(),
            
            # Estadísticas de trades
            'largest_win': winning_trades['profit_loss'].max(),
            'largest_loss': losing_trades['profit_loss'].min(),
            'average_win': winning_trades['profit_loss'].mean(),
            'average_loss': losing_trades['profit_loss'].mean(),
            
            # Análisis temporal
            'average_trade_duration': self._calculate_average_trade_duration(),
            'trades_per_day': self._calculate_trades_per_day()
        }
        
        return metrics
    
    def _calculate_max_drawdown(self) -> float:
        """
        Calcula el máximo drawdown
        
        :return: Máximo drawdown en porcentaje
        """
        cumulative_returns = self.calculate_returns()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        return drawdown.min() * 100
    
    def _calculate_sharpe_ratio(
        self, 
        risk_free_rate: float = 0.02
    ) -> float:
        """
        Calcula el ratio de Sharpe
        
        :param risk_free_rate: Tasa libre de riesgo
        :return: Ratio de Sharpe
        """
        returns = self.trades_df['profit_loss'] / 100
        excess_returns = returns - (risk_free_rate / 252)  # Ajuste diario
        
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
    def _calculate_sortino_ratio(
        self, 
        risk_free_rate: float = 0.02
    ) -> float:
        """
        Calcula el ratio de Sortino
        
        :param risk_free_rate: Tasa libre de riesgo
        :return: Ratio de Sortino
        """
        returns = self.trades_df['profit_loss'] / 100
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf')
        
        excess_returns = returns.mean() - (risk_free_rate / 252)
        downside_deviation = downside_returns.std()
        
        return excess_returns / downside_deviation * np.sqrt(252)
    
    def _calculate_average_trade_duration(self) -> pd.Timedelta:
        """
        Calcula la duración promedio de los trades
        
        :return: Duración promedio
        """
        if 'exit_timestamp' in self.trades_df.columns and 'timestamp' in self.trades_df.columns:
            durations = self.trades_df['exit_timestamp'] - self.trades_df['timestamp']
            return durations.mean()
        return pd.Timedelta(0)
    
    def _calculate_trades_per_day(self) -> float:
        """
        Calcula el número de trades por día
        
        :return: Trades por día
        """
        days = (self.trades_df['timestamp'].max() - self.trades_df['timestamp'].min()).days
        return len(self.trades_df) / max(1, days)
    
    def plot_equity_curve(self, save_path: Optional[str] = None):
        """
        Genera gráfico de curva de equity
        
        :param save_path: Ruta para guardar el gráfico
        """
        cumulative_returns = self.calculate_returns()
        
        plt.figure(figsize=(12, 6))
        plt.plot(cumulative_returns.index, cumulative_returns.values * 100)
        plt.title('Curva de Equity')
        plt.xlabel('Fecha')
        plt.ylabel('Retorno Acumulado (%)')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def plot_profit_distribution(self, save_path: Optional[str] = None):
        """
        Genera histograma de distribución de ganancias/pérdidas
        
        :param save_path: Ruta para guardar el gráfico
        """
        plt.figure(figsize=(12, 6))
        sns.histplot(self.trades_df['profit_loss'], kde=True)
        plt.title('Distribución de Ganancias/Pérdidas')
        plt.xlabel('Profit/Loss')
        plt.ylabel('Frecuencia')
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def generate_detailed_report(
        self, 
        output_path: Optional[str] = None
    ) -> Dict:
        """
        Genera un informe detallado de rendimiento
        
        :param output_path: Ruta para guardar el informe
        :return: Diccionario con informe de rendimiento
        """
        metrics = self.calculate_performance_metrics()
        
        # Generar gráficos
        self.plot_equity_curve(
            save_path=f"{output_path}/equity_curve.png" if output_path else None
        )
        self.plot_profit_distribution(
            save_path=f"{output_path}/profit_distribution.png" if output_path else None
        )
        
        return metrics

# Ejemplo de uso
def main():
    # Datos de ejemplo de trades
    sample_trades = [
        {
            'timestamp': pd.Timestamp('2023-01-01'),
            'exit_timestamp': pd.Timestamp('2023-01-02'),
            'profit_loss': 1.5,
            'symbol': 'BTC/USDT'
        },
        {
            'timestamp': pd.Timestamp('2023-01-03'),
            'exit_timestamp': pd.Timestamp('2023-01-05'),
            'profit_loss': -0.7,
            'symbol': 'BTC/USDT'
        }
    ]
    
    # Inicializar analizador
    analyzer = PerformanceAnalyzer(sample_trades)
    
    # Generar informe detallado
    report = analyzer.generate_detailed_report(output_path='./performance_reports')
    
    # Imprimir métricas
    for metric, value in report.items():
        print(f"{metric}: {value}")

if __name__ == "__main__":
    main()
