# Otimizador de Hiperparâmetros para YOLO com Ray Tune

Este repositório fornece um pipeline de MLOps completo, robusto e orientado por configuração para encontrar os hiperparâmetros ideais para um modelo de detecção de objetos YOLO. O projeto utiliza Ray Tune para a otimização e segue uma metodologia de avaliação padrão-ouro para garantir resultados confiáveis.

O processo inteiro é automatizado, desde a preparação dos dados e a busca por hiperparâmetros até o treinamento do modelo final e sua avaliação imparcial em um conjunto de teste cego.

---

## 📋 Funcionalidades Principais

-   **HPO Automatizado**: Utiliza o **Ray Tune** com o algoritmo de busca **Optuna** para uma otimização de hiperparâmetros eficiente, baseada em Otimização Bayesiana.
-   **Metodologia Robusta**: Emprega **Validação Cruzada K-Fold** durante a fase de HPO para garantir que os hiperparâmetros escolhidos sejam robustos e não superajustados a uma única divisão de dados. A **mediana do mAP de validação** entre os folds é usada como a métrica principal.
-   **Avaliação Imparcial (Padrão-Ouro)**: Segue estritamente a prática de reservar um **conjunto de teste final cego** (`final test set`), que é utilizado apenas uma vez para avaliar o modelo de produção final.
-   **Orientado por Configuração**: Todo o pipeline é controlado por um arquivo central `config.yaml`, facilitando a modificação de caminhos, divisão de datasets e parâmetros de treinamento sem tocar no código-fonte.
-   **Reprodutibilidade**: Utiliza uma semente aleatória (`random_seed`) fixa para a divisão dos dados, garantindo a reprodutibilidade dos experimentos.
-   **Código Modular e Limpo**: A estrutura do projeto separa as responsabilidades, com docstrings e type hints, tornando o código fácil de entender, manter e estender.

---

## 📁 Estrutura do Projeto

```
/yolo_tuner_project/
|
|-- config.yaml                 # ⚙️ Arquivo de configuração central para todo o pipeline
|-- main.py                     # ▶️ Script orquestrador principal para rodar tudo
|-- README.md                   # 📄 Este arquivo
|-- requirements.txt            # 📦 Dependências Python
|
|-- base_models/                # 🤖 Coloque seus modelos .pt pré-treinados aqui (ex: yolov8n.pt)
|
|-- data/
|   |-- full_dataset/           # 📥 COLOQUE SEU DATASET COMPLETO AQUI
|   |   |-- dataset.yaml        # ❗ IMPORTANTE: Defina suas classes aqui
|   |   |-- images/
|   |   |-- labels/
|   |
|   |-- dev_dataset/            # (Gerado) Para HPO e treinamento final
|   |-- final_test_dataset/     # (Gerado) O conjunto de teste cego
|
|-- scripts/
|   |-- prepare_dataset.py      # Script para a preparação e divisão dos dados
|
|-- src/                        # Código-fonte com toda a lógica do pipeline
|
|-- production_model/           # (Gerado) O modelo final treinado é salvo aqui
|-- ray_results/                # (Gerado) Logs detalhados e artefatos do Ray Tune
```

---

## 🚀 Configuração e Instalação

Siga estes passos para configurar o ambiente e preparar seus dados.

### 1. Clone o Repositório
```bash
git clone <url-do-seu-repositorio>
cd yolo_tuner_project
```

### 2. Instale as Dependências
É altamente recomendado usar um ambiente virtual (`venv`).

```bash
# Crie e ative o ambiente virtual
python -m venv venv
source venv/bin/activate  # No Windows, use: venv\Scripts\activate

# Instale os pacotes
pip install -r requirements.txt
```

### 3. Adicione os Modelos Base
Baixe os arquivos `.pt` dos modelos YOLO que você deseja incluir na busca (ex: `yolov8n.pt`, `yolov9s.pt`) e coloque-os dentro da pasta `base_models/`.

### 4. Prepare seu Dataset
Esta é a etapa mais importante da configuração.

-   Coloque **todas** as suas imagens de desenvolvimento na pasta `data/full_dataset/images/`.
-   Coloque **todas** as suas anotações (labels) correspondentes na pasta `data/full_dataset/labels/`.
-   **CRUCIAL:** Crie um arquivo chamado `data/full_dataset/dataset.yaml` para descrever suas classes. Ele deve seguir este formato:

    ```yaml
    # Este arquivo informa o pipeline sobre as suas classes.
    # Os caminhos são apenas para referência e não são usados pelos scripts.
    train: ./images/
    val: ./images/

    # --- EDITE ESTA SEÇÃO ---
    nc: 3
    names: ['Californicus', 'Macropilis', 'Rajado']
    ```

---

## 🛠️ Uso - Executando o Pipeline

Todo o pipeline é controlado pelo `config.yaml` e executado com um único comando.

### 1. Personalize a Configuração
Abra o arquivo `config.yaml` e revise os parâmetros para adequá-los ao seu projeto e hardware. Os mais importantes são:

-   `dataset.test_split_ratio`: A porcentagem de dados a ser reservada para o teste final (ex: `0.1` para 10%).
-   `hpo.num_samples`: O número total de combinações de hiperparâmetros a serem testadas. Mais amostras podem gerar melhores resultados, mas levarão mais tempo.
-   `hpo.resources_per_trial`: Ajuste a alocação de `cpu` e `gpu` com base na sua máquina.
-   `training.epochs`: O número de épocas para treinar em cada fold durante a fase de HPO.

### 2. Execute o Pipeline Completo
No terminal, a partir da pasta raiz do projeto, execute o script principal:

```bash
python main.py
```

O script irá orquestrar todas as etapas em sequência:
1.  **Etapa 0: Preparar Datasets**: Divide os dados e gera os arquivos YAML necessários.
2.  **Etapa 1: Otimização de Hiperparâmetros (HPO)**: Roda o Ray Tune. Esta é a etapa mais demorada.
3.  **Etapa 2: Treinar Modelo Final**: Treina um novo modelo do zero usando os melhores hiperparâmetros encontrados.
4.  **Etapa 3: Avaliação Final**: Avalia o modelo final no conjunto de teste cego e imprime os resultados no console.

### 3. Analise os Resultados
-   **Melhores Hiperparâmetros**: A configuração ideal é salva no arquivo `best_config.json`.
-   **Modelo Final**: O modelo treinado e pronto para produção é salvo na pasta `production_model/`.
-   **Detalhes da HPO**: Para uma análise visual e aprofundada do desempenho de cada tentativa, utilize o TensorBoard:

    ```bash
    tensorboard --logdir ray_results
    ```

---

## 🔬 Visão Geral da Metodologia (O Fluxo "Padrão-Ouro")

Este projeto segue um processo rigoroso para garantir que os resultados sejam robustos, confiáveis e generalizáveis.

1.  **Divisão dos Dados**: O dataset inicial (`full_dataset`) é dividido em um **Conjunto de Desenvolvimento** (ex: 90%) e um **Conjunto de Teste Final** (ex: 10%). O Conjunto de Teste Final é imediatamente "trancado" e não é utilizado até o final.

2.  **Otimização de Hiperparâmetros (HPO)**: O Ray Tune opera **exclusivamente** no Conjunto de Desenvolvimento. Para cada combinação de hiperparâmetros que ele testa, uma Validação Cruzada K-Fold completa é realizada. A **mediana do mAP de validação** entre os folds é usada para julgar a qualidade da configuração, tornando a busca robusta a resultados anômalos.

3.  **Treinamento do Modelo Final**: Uma vez que a melhor configuração é encontrada, um único e novo modelo é treinado do zero usando **100% dos dados do Conjunto de Desenvolvimento**.

4.  **Avaliação Imparcial**: Este modelo final é então avaliado **uma única vez** no Conjunto de Teste Final. As métricas desta avaliação (mAP, F1, Precision, Recall) são reportadas como o verdadeiro desempenho do sistema.

Essa separação previne o vazamento de dados (`data leakage`) e garante que as métricas de desempenho finais sejam uma estimativa honesta da capacidade do modelo de generalizar para novos dados nunca antes vistos.