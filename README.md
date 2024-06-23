# Fluctuaciones del Bitcoin
El valor del Bitcoin fluctúa diariamente, por lo que se realizara una predicción sobre el valor que tomara este bien en el mercado, teniendo en cuenta un periodo económico sin eventos excepcionales. Se aplicaran métodos de análisis de series temporales, procedimiento ARIMA y aprendizaje automático.
## Introducción
Bitcoin ha capturado la imaginación del mundo financiero y del público en general, no solo como un medio innovador de intercambio sino también como un activo altamente volátil y especulativo. Su precio ha visto oscilaciones dramáticas, influenciado por una amplia gama de factores que van desde desarrollos regulatorios hasta la adopción por parte de instituciones financieras y la percepción pública general. Entender y predecir estas fluctuaciones es de gran interés para inversores, traders y analistas de políticas, pero la tarea es complicada por la naturaleza compleja y a menudo opaca del mercado de criptomonedas. Este proyecto busca aplicar y evaluar técnicas avanzadas de aprendizaje automático para prever la volatilidad del precio de Bitcoin y dilucidar los factores subyacentes que impulsan estos cambios.
## ¿Qué son las series temporales?​
Las series temporales son conjuntos de datos que se recopilan o registran en intervalos de tiempo regulares. Estos datos se ordenan cronológicamente y se utilizan para analizar la evolución, tendencias y patrones que pueden estar presentes en un fenómeno específico a lo largo del tiempo.​
## ¿Por qué usamos series de tiempo?​
Captura de Cambios Temporales: Las series de tiempo nos permiten capturar los cambios en los datos a lo largo del tiempo. En el caso del Bitcoin, su volatilidad puede cambiar drásticamente de un día para otro, y las series de tiempo nos ayudaran a visualizar y entender estos cambios.​

Predicción y Proyección: Una vez que hayamos analizado y comprendido los datos históricos de la volatilidad del Bitcoin, las series de tiempo nos permitirán construir modelos predictivos para proyectar futuros movimientos de volatilidad. Esto puede ser útil para la toma de decisiones en inversiones.​

Las series temporales pueden ser analizadas utilizando diversos métodos y técnicas. Nosotros usaremos el modelo ARIMA, en particular el modelo ARIMA(2,1,3) with drift es un modelo de series temporales que incluye un componente de deriva. Este modelo es una extensión del modelo ARIMA estándar que también tiene términos autorregresivos, integrados y de medias móviles. El código mediante la función auto.arima es el que nos arroja que el modelo a utilizar debe de ser el ARIMA(2,1,3).​

Usaremos la función "forecast", ya que es una herramienta poderosa para realizar pronósticos en series temporales y puede ser muy útil al momento de analizar la volatilidad del Bitcoin utilizando el modelo ARIMA.
## Análisis de Residuos
El análisis de residuos de un modelo estadístico es una parte esencial en el proceso de modelado de series temporales, especialmente cuando usas modelos ARIMA para predecir precios de criptomonedas como el Bitcoin​.

Usaremos residuos ya que al centrarnos solamente en datos y no datos exógenos, se presentan residuos marcados.​

## ¿Qué son los Residuos?

Los residuos en un modelo estadístico son las diferencias entre los valores observados de la serie temporal y los valores ajustados que el modelo predice. En el contexto de un modelo ARIMA, los residuos son esencialmente los errores de predicción del modelo para cada punto en el tiempo.​

## ¿Para qué Sirve el Análisis de Residuos?​

Verificar Ajuste del Modelo: Diagnóstico de Independencia: Idealmente, los residuos de un modelo bien ajustado deben comportarse como "ruido blanco". Esto significa que los residuos deben ser independientes entre sí (sin autocorrelaciones) y estar distribuidos aleatoriamente alrededor de cero con varianza constante.​

Normalidad: A menudo se busca que los residuos sigan una distribución normal, lo que sugiere que el modelo ha capturado toda la información en los datos, dejando solo el "ruido" aleatorio.​

Identificación de Modelos Inadecuados: Si los residuos muestran patrones, tendencias, o autocorrelación significativa, esto puede indicar que el modelo no ha capturado alguna dinámica subyacente en los datos, lo que podría llevar a reconsiderar el tipo de modelo utilizado, los parámetros del modelo, o la inclusión de términos adicionales o variables exógenas.​

## Garch

Para obtener una visión más completa y robusta del mercado, se hizo la comparación con otro modelo, específicamente GARCH. El modelo GARCH (Generalized Autoregressive Conditional Heteroskedasticity) es una extensión del modelo ARCH (Autoregressive Conditional Heteroskedasticity) y está diseñado específicamente para modelar y pronosticar la volatilidad condicional en series temporales financieras. Se puede decir que es un enfoque para estimar la volatilidad de los mercados financieros . Las instituciones financieras utilizan el modelo para estimar la volatilidad del rendimiento de acciones, bonos y otros vehículos de inversión.

Finalmente obtenemos que el valor de los p-values son menores a 0,05, por lo que nuestros resultados son estadísticamente significativos. ¿Qué significa esto?, un p-value menor a 0,05 indica que la probabilidad de que los resultados observados se deban al azar es menor al 5%, lo cual nos proporciona una base sólida para poder avanzar con confianza en nuestras conclusiones.​

# R Notebook

library(forecast)

library(quantmod)

library(rugarch)

### Carga los datos históricos del Bitcoin (por ejemplo, utilizando el paquete quantmod)

btc_data <- getSymbols("BTC-USD", from="2015-01-01", src = "yahoo", auto.assign = FALSE)

plot(btc_data, main = "Historical Bitcoin Data", xlab = "Date", ylab = "Price (USD)")

print(btc_data)

btc_close <- Cl(btc_data)  

### Obtiene los precios de cierre

plot(btc_close, main = "Bitcoin Closing Prices", xlab = "Date", ylab = "Closing Price (USD)")

### Convierte los datos a una serie temporal

btc_ts <- ts(btc_close, frequency = 1)

print(btc_ts)

### Ajusta el modelo ARIMA automáticamente

btc_arima <- auto.arima(btc_ts)

print(btc_arima)

### Análisis de residuos

checkresiduals(btc_arima)

### Realiza el pronóstico

btc_forecast <- forecast(btc_arima, h = 30)  # Pronóstico para los próximos 30 periodos

### Imprime el pronóstico

print(btc_forecast)

### Grafica el pronóstico con un "zoom" en la predicción
plot(btc_forecast, xlim = c(length(btc_ts) - 100, length(btc_ts) + 30), 
     main = "Forecasts from ARIMA(3,1,2) with drift",
     xlab = "Time", ylab = "Price (USD)")

### Calcular los rendimientos logarítmicos
btc_returns <- diff(log(btc_close))

btc_returns <- btc_returns[!is.na(btc_returns)]  # Eliminar NA resultantes de la diferenciación

### Configurar el modelo GARCH (por ejemplo, GARCH(1,1))
spec <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
                   mean.model = list(armaOrder = c(0, 0), include.mean = TRUE),
                   distribution.model = "norm")

### Ajustar el modelo a los datos
fit <- ugarchfit(spec = spec, data = btc_returns)

### Imprimir el resumen del ajuste
print(fit)

### Visualizar los resultados del modelo GARCH
plot(fit, which = "all")
