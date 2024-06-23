# Fluctuaciones-del-Bitcoin
El valor del Bitcoin fluctúa diariamente, por lo que se realizara una predicción sobre el valor que tomara este bien en el mercado, teniendo en cuenta un periodo económico sin eventos excepcionales. Se aplicaran métodos de análisis de series temporales, procedimiento ARIMA y aprendizaje automático.
# Introducción
Bitcoin ha capturado la imaginación del mundo financiero y del público en general, no solo como un medio innovador de intercambio sino también como un activo altamente volátil y especulativo. Su precio ha visto oscilaciones dramáticas, influenciado por una amplia gama de factores que van desde desarrollos regulatorios hasta la adopción por parte de instituciones financieras y la percepción pública general. Entender y predecir estas fluctuaciones es de gran interés para inversores, traders y analistas de políticas, pero la tarea es complicada por la naturaleza compleja y a menudo opaca del mercado de criptomonedas. Este proyecto busca aplicar y evaluar técnicas avanzadas de aprendizaje automático para prever la volatilidad del precio de Bitcoin y dilucidar los factores subyacentes que impulsan estos cambios.
# ¿Qué son las series temporales?​
Las series temporales son conjuntos de datos que se recopilan o registran en intervalos de tiempo regulares. Estos datos se ordenan cronológicamente y se utilizan para analizar la evolución, tendencias y patrones que pueden estar presentes en un fenómeno específico a lo largo del tiempo.​
# ¿Por qué usamos series de tiempo?​
Captura de Cambios Temporales: Las series de tiempo nos permiten capturar los cambios en los datos a lo largo del tiempo. En el caso del Bitcoin, su volatilidad puede cambiar drásticamente de un día para otro, y las series de tiempo nos ayudaran a visualizar y entender estos cambios.​

Predicción y Proyección: Una vez que hayamos analizado y comprendido los datos históricos de la volatilidad del Bitcoin, las series de tiempo nos permitirán construir modelos predictivos para proyectar futuros movimientos de volatilidad. Esto puede ser útil para la toma de decisiones en inversiones.​

Las series temporales pueden ser analizadas utilizando diversos métodos y técnicas. Nosotros usaremos el modelo ARIMA, en particular el modelo ARIMA(2,1,3) with drift es un modelo de series temporales que incluye un componente de deriva. Este modelo es una extensión del modelo ARIMA estándar que también tiene términos autorregresivos, integrados y de medias móviles. El código mediante la función auto.arima es el que nos arroja que el modelo a utilizar debe de ser el ARIMA(2,1,3).​

Usaremos la función "forecast", ya que es una herramienta poderosa para realizar pronósticos en series temporales y puede ser muy útil al momento de analizar la volatilidad del Bitcoin utilizando el modelo ARIMA.
