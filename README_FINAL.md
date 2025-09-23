# 🚦 App de Detecção de Placas de Trânsito - YOLOv8

Aplicativo Streamlit profissional para detecção de placas de trânsito em tempo real usando YOLOv8.

## 🎯 Visão Geral

Este projeto implementa um sistema completo de detecção de placas de trânsito com:
- **4 Classes**: Limite de velocidade, faixa de pedestres, semáforo e placa de parada
- **Modelo YOLOv8**: Arquitetura nano otimizada para velocidade
- **Interface Profissional**: Design moderno e responsivo
- **Análise Completa**: Gráficos interativos de treinamento e métricas

## 🚀 Instalação e Execução

### Opção 1: Execução Rápida
```bash
# Clone o repositório
git clone <repository-url>
cd road_sign_detection_yolo

# Execute o script automático
./run_app.sh
```

### Opção 2: Instalação Manual
```bash
# 1. Crie ambiente virtual
python -m venv venv
source venv/bin/activate

# 2. Instale dependências
pip install -r requirements_lite.txt

# 3. Execute o app
streamlit run app_lite.py
```

## 📱 Funcionalidades

### 🔍 Detecção de Imagens
- Upload de imagens (PNG, JPG, JPEG)
- Detecção em tempo real com bounding boxes
- Ajuste de limiar de confiança
- Tabela de resultados com confiança

### 📊 Análise do Modelo
- Métricas de performance em tempo real
- Informações detalhadas das classes
- Estatísticas do dataset

### 📈 Visualização de Treinamento
- Gráficos interativos com Plotly
- Curvas de métricas (Precision, Recall, mAP)
- Análise de loss functions
- Estatísticas detalhadas do processo

### ℹ️ Documentação
- Informações completas do projeto
- Tecnologias utilizadas
- Performance do modelo

## 🏗️ Arquitetura

```
road_sign_detection_yolo/
├── app.py                    # App completo com YOLO
├── app_lite.py              # App demo (sem YOLO)
├── requirements.txt         # Dependências completas
├── requirements_lite.txt    # Dependências mínimas
├── run_app.sh              # Script de execução
├── demo.py                 # Verificação de dependências
├── dados/
│   ├── road_signs_annotations.csv
│   └── road_signs_dataset.yaml
├── dataset/
│   ├── train/ (701 imagens)
│   └── val/ (176 imagens)
├── modelos/
│   ├── best.pt
│   └── last.pt
├── resultados/
│   └── runs/detect/train/
└── notebooks/
    ├── 1_Exploratory_Data_Analysis_.ipynb
    ├── 2_Data_Pre_Processing.ipynb
    └── 3_Model_Training.ipynb
```

## 📊 Dataset

- **Total**: 1.244 anotações de sinais
- **Treino**: 701 imagens (984 anotações)
- **Validação**: 176 imagens (260 anotações)
- **Classes**:
  - 🚦 Limite de Velocidade: 783 anotações
  - 🚶 Faixa de Pedestres: 200 anotações
  - 🔴 Semáforo: 170 anotações
  - 🛑 Placa de Parada: 91 anotações

## 📈 Performance

- **mAP@0.5**: ~92%
- **mAP@0.5:0.95**: ~77%
- **Precision**: ~94%
- **Recall**: ~88%

## 🛠️ Tecnologias

### Backend
- **Python 3.13**
- **Streamlit** - Interface web
- **OpenCV** - Processamento de imagens
- **PIL/Pillow** - Manipulação de imagens

### Machine Learning
- **YOLOv8** - Detecção de objetos
- **Ultralytics** - Framework YOLO
- **PyTorch** - Backend ML (opcional)

### Visualização
- **Plotly** - Gráficos interativos
- **Pandas** - Manipulação de dados
- **NumPy** - Computação numérica

## 🎨 Interface

### Design Profissional
- Cores corporativas (azul e gradientes)
- Layout responsivo
- Componentes interativos
- Feedback visual em tempo real

### Navegação
- Sistema de abas organizado
- Sidebar com configurações
- Métricas em tempo real
- Gráficos interativos

## 🔧 Configurações

### Limiar de Confiança
- Range: 0.1 - 1.0
- Padrão: 0.5
- Ajustável em tempo real

### Formatos Suportados
- PNG, JPG, JPEG
- Resolução automática
- Otimização para 640x640

## 📝 Versões

### app.py (Completo)
- Requer YOLOv8 instalado
- Detecção real com modelo treinado
- Dependências: PyTorch, Ultralytics

### app_lite.py (Demo)
- Simulação de detecções
- Sem dependências pesadas
- Ideal para demonstração

## 🚀 Deploy

### Streamlit Cloud
1. Conecte o repositório
2. Configure `requirements_lite.txt`
3. Execute `streamlit run app_lite.py`

### Local
```bash
./run_app.sh
```

## 📖 Documentação

- **README_FINAL.md** - Este arquivo
- **README_app.md** - Documentação do app
- **notebooks/** - Processo de desenvolvimento
- **demo.py** - Verificação de dependências

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch (`git checkout -b feature/nova-funcionalidade`)
3. Commit suas mudanças (`git commit -am 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/nova-funcionalidade`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## 👨‍💻 Autor

Desenvolvido com ❤️ usando Streamlit e YOLOv8

---

**🎉 Pronto para usar! Execute `./run_app.sh` e comece a detectar placas de trânsito!**
