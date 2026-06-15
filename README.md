# water-quality-ia
Repositório dedicado ao Trabalho final da matéria Inteligencia Artificial, 2026. Classificação de potabilidade da água utilizando Naive Bayes e Rede Neural Artificial.

# Disciplina de Inteligência Artificial, Professor Munif, Unicesumar 2026

## Integrantes

- Vinicius Reginaldo Ferrarini — RA: 23159293-2 
- Pedro Alvaro Mantuani Silva  — RA: 23079477-2 
- Luiz Henrique Peschieira Romano  — RA: 25363238-2

---

## Resumo do Projeto

### Contextualização

O acesso à água potável é reconhecido pela Organização Mundial da Saúde (OMS) como um direito humano fundamental. Apesar disso, milhões de pessoas ao redor do mundo ainda consomem água contaminada, o que resulta em doenças graves e mortes evitáveis. A análise laboratorial tradicional da qualidade da água é eficaz, porém lenta e cara. Nesse contexto, o uso de Inteligência Artificial surge como uma alternativa promissora para automatizar e agilizar o processo de classificação da potabilidade da água com base em parâmetros físico-químicos.

### Problema

O objetivo deste projeto é desenvolver e comparar modelos de Inteligência Artificial capazes de **classificar amostras de água como potáveis ou não potáveis**, utilizando métricas físico-químicas disponíveis no dataset "Water Quality". A questão central é: *é possível prever, com confiança, se uma amostra de água é segura para consumo humano a partir de dados laboratoriais?*

### Hipótese

Nossa hipótese é que **modelos de IA conseguem classificar a potabilidade da água com acurácia satisfatória** a partir de atributos físico-químicos, e que a Rede Neural Artificial (RNA) tende a superar o Naive Bayes nessa tarefa, dado que a RNA é capaz de capturar relações não-lineares mais complexas entre os atributos.

---

## Dataset

### Descrição

- **Nome:** Water Quality (Potability)
- **Origem:** [Kaggle - Water Quality](https://www.kaggle.com/datasets/mssmartypants/water-quality)
- **Quantidade de registros:** 7.999 amostras
- **Formato:** CSV tabular

### Atributos

| Atributo | Descrição |
|---|---|
| `aluminium` | Concentração de alumínio (mg/L) |
| `ammonia` | Concentração de amônia (mg/L) |
| `arsenic` | Concentração de arsênico (mg/L) |
| `barium` | Concentração de bário (mg/L) |
| `cadmium` | Concentração de cádmio (mg/L) |
| `chloramine` | Concentração de cloramina (mg/L) |
| `chromium` | Concentração de cromo (mg/L) |
| `copper` | Concentração de cobre (mg/L) |
| `flouride` | Concentração de fluoreto (mg/L) |
| `bacteria` | Presença de bactérias (0 ou 1) |
| `viruses` | Presença de vírus (0 ou 1) |
| `lead` | Concentração de chumbo (mg/L) |
| `nitrates` | Concentração de nitratos (mg/L) |
| `nitrites` | Concentração de nitritos (mg/L) |
| `mercury` | Concentração de mercúrio (mg/L) |
| `perchlorate` | Concentração de perclorato (mg/L) |
| `radium` | Concentração de rádio (pCi/L) |
| `selenium` | Concentração de selênio (mg/L) |
| `silver` | Concentração de prata (mg/L) |
| `uranium` | Concentração de urânio (mg/L) |
| `is_safe` | **Variável alvo** — 1 = potável, 0 = não potável |

### Variável Alvo

A variável alvo é `is_safe`, que indica se a amostra de água é segura para consumo humano (`1`) ou não (`0`). Trata-se, portanto, de um problema de **classificação binária**.

### Preparação dos Dados

1. **Carregamento:** leitura do arquivo CSV com a biblioteca `pandas`.
2. **Tratamento de valores ausentes:** verificação e remoção de linhas com valores `NaN`, garantindo integridade do dataset.
3. **Separação de features e target:** variáveis independentes (X) separadas da variável alvo `is_safe` (y).
4. **Normalização:** aplicação do `StandardScaler` do `scikit-learn` para padronizar os atributos numéricos (média 0, desvio padrão 1), etapa necessária especialmente para a RNA.
5. **Divisão treino/teste:** divisão em 80% para treino e 20% para teste, usando `train_test_split` com `random_state=42` para reprodutibilidade.

---

## Métodos de Inteligência Artificial

### Método 1 — Naive Bayes (Parte 1 da disciplina)

O **Naive Bayes Gaussiano** é um classificador probabilístico fundamentado no Teorema de Bayes. Ele assume que todas as features são **independentes entre si** (hipótese "naive") e que seguem uma distribuição gaussiana. Apesar de sua simplicidade, é amplamente utilizado em problemas de classificação e serve como boa baseline.

**Implementação:**
```python
from sklearn.naive_bayes import GaussianNB

modelo_nb = GaussianNB()
modelo_nb.fit(X_train, y_train)
y_pred_nb = modelo_nb.predict(X_test)
```

### Método 2 — Rede Neural Artificial — RNA (Parte 2 da disciplina)

A **Rede Neural Artificial** é um modelo inspirado no funcionamento do cérebro humano, composto por camadas de neurônios artificiais (nós) que aprendem padrões complexos e não-lineares nos dados.

A arquitetura utilizada foi:
- **Camada de entrada:** 20 neurônios (um por feature)
- **1ª camada oculta:** 64 neurônios com ativação ReLU
- **2ª camada oculta:** 32 neurônios com ativação ReLU
- **Camada de saída:** 1 neurônio com ativação Sigmoid (saída probabilística 0–1)

**Implementação:**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

modelo_rna = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

modelo_rna.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

historico = modelo_rna.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)
```

---

## Avaliação dos Modelos

### Métricas utilizadas

Para avaliar o desempenho dos modelos, foram utilizadas as seguintes métricas padrão para classificação binária:

- **Acurácia:** proporção de classificações corretas sobre o total.
- **Precisão:** dos exemplos classificados como positivos, quantos realmente são.
- **Revocação (Recall):** dos exemplos positivos reais, quantos foram corretamente identificados.
- **F1-Score:** média harmônica entre precisão e revocação.
- **Matriz de Confusão:** visualização dos acertos e erros por classe.
- **Curva de perda e acurácia (RNA):** acompanhamento do aprendizado ao longo das épocas.

### Resultados — Naive Bayes

```
=== Resultados — Naive Bayes ===
Acurácia:  0.8444 (84.44%)
Precisão:  0.4169
Recall:    0.6150
F1-Score:  0.4970

=== Relatório Completo ===
              precision    recall  f1-score   support

 Não Potável       0.94      0.88      0.91      1400
     Potável       0.42      0.61      0.50       200

    accuracy                           0.84      1600
   macro avg       0.68      0.75      0.70      1600
weighted avg       0.88      0.84      0.86      1600
```

### Resultados — Rede Neural Artificial

```
=== Resultados — Rede Neural Artificial ===
Acurácia:  0.9456 (94.56%)
Precisão:  0.8266
Recall:    0.7150
F1-Score:  0.7668

=== Relatório Completo ===
              precision    recall  f1-score   support

 Não Potável       0.96      0.98      0.97      1400
     Potável       0.83      0.71      0.77       200

    accuracy                           0.95      1600
   macro avg       0.89      0.85      0.87      1600
weighted avg       0.94      0.95      0.94      1600
```

### Gráficos de Avaliação

> 1. ![Matriz de Confusão - Naive Bayes](graficos/matriz_confusao_nb.png)
> 2. ![Matriz de Confusão - RNA](graficos/matriz_confusao_rna.png)
> 3. ![Curva de Acurácia - RNA](graficos/curva_acuracia_rna.png)
> 4. ![Curva de Perda - RNA](graficos/curva_perda_rna.png)

---

## Comparação dos Resultados

| Métrica | Naive Bayes | Rede Neural (RNA) |
|---|---|---|
| Acurácia | 84% | 94% |
| Precisão | 41% | 82% |
| Recall | 61% | 71% |
| F1-Score | 49% | 76% |

![Comparativo dos Modelos](graficos/comparativo_modelos.png)

### Análise Comparativa

O Naive Bayes, por assumir independência entre as features e distribuição gaussiana dos dados, pode perder desempenho em situações onde há correlação entre os atributos físico-químicos. No contexto de qualidade da água, é razoável esperar que variáveis como concentração de metais pesados e presença de bactérias possam apresentar dependências entre si, o que limita a eficácia do modelo.

A Rede Neural Artificial, por sua vez, é capaz de aprender representações hierárquicas e relações não-lineares entre as variáveis, o que tende a resultar em melhor capacidade de generalização para problemas complexos como este. O custo é um maior tempo de treinamento e necessidade de ajuste de hiperparâmetros.

A comparação final entre os modelos foi baseada principalmente na **acurácia** e no **F1-Score**, pois em um contexto de potabilidade da água, tanto os falsos positivos (classificar água contaminada como segura) quanto os falsos negativos (descartar água segura) têm impacto relevante.

---

## Conclusão

Este trabalho demonstrou o processo completo de desenvolvimento de uma solução baseada em Inteligência Artificial para o problema de classificação da potabilidade da água. Foram aplicados dois modelos distintos: o **Naive Bayes Gaussiano** (Parte 1 da disciplina) e a **Rede Neural Artificial** (Parte 2 da disciplina), ambos treinados e avaliados sobre o mesmo dataset.

A escolha do dataset "Water Quality" se mostrou adequada ao problema, por conter atributos físico-químicos reais e uma variável alvo bem definida (potável ou não potável), tornando o problema diretamente aplicável ao contexto de saúde pública e ambiental.

Os resultados obtidos confirmaram a hipótese inicial: a RNA alcançou 94,44% de acurácia contra 84,44% do Naive Bayes, demonstrando maior capacidade de capturar as relações não-lineares entre os atributos físico-químicos da água. O F1-Score da RNA (0,76) também superou significativamente o do Naive Bayes (0,49), indicando melhor equilíbrio entre precisão e recall.

---

## Estrutura do Repositório

```
📁 water-quality-ia/
│
├── 📄 README.md                  # Documentação principal
├── 📄 README.pdf                 # PDF com o conteúdo do README
├── 📄 water_quality.ipynb        # Notebook Jupyter com todo o código
├── 📄 water_quality.csv          # Dataset (ou ver instruções abaixo)
├── 📁 modelos/
│   ├── modelo_naive_bayes.pkl    # Modelo Naive Bayes treinado
│   └── modelo_rna.h5             # Modelo RNA treinado
└── 📁 graficos/
    ├── matriz_confusao_nb.png
    ├── matriz_confusao_rna.png
    ├── curva_acuracia_rna.png
    ├── curva_perda_rna.png
    └── comparativo_modelos.png
```

### Como obter o dataset

1. Acesse: https://www.kaggle.com/datasets/mssmartypants/water-quality
2. Faça o download do arquivo `waterQuality1.csv`
3. Renomeie para `water_quality.csv` e coloque na raiz do projeto

### Como executar o projeto

```bash
# Instalar dependências
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn joblib

# Executar o notebook
jupyter notebook water_quality.ipynb
```

### Como carregar os modelos treinados

```python
import joblib
from tensorflow.keras.models import load_model

# Naive Bayes
modelo_nb = joblib.load('modelos/modelo_naive_bayes.pkl')

# RNA
modelo_rna = load_model('modelos/modelo_rna.h5')
```
---

## Dependências

```
pandas
numpy
scikit-learn
tensorflow
matplotlib
seaborn
joblib
jupyter
```
---
*Disciplina de Inteligência Artificial — Professor Munif Gebara Junior — Unicesumar 2026*

