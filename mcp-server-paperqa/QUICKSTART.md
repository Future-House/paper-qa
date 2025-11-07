# Guide de démarrage rapide - PaperQA MCP Server

## Installation en 5 minutes

### 1. Installer le serveur

```bash
cd mcp-server-paperqa
./install.sh
```

Le script vous demandera :
- Le chemin vers votre dossier de papers (défaut : `~/papers`)
- Votre clé API OpenAI (optionnel si déjà définie)

### 2. Ajouter des papers

Copiez vos PDFs dans le dossier papers :

```bash
cp ~/Downloads/*.pdf ~/papers/
```

Ou téléchargez depuis arXiv :

```bash
cd ~/papers
wget https://arxiv.org/pdf/2301.12345.pdf
```

### 3. Redémarrer Claude Code

Fermez et relancez Claude Code pour charger la configuration MCP.

### 4. Utiliser PaperQA

Ouvrez Claude Code et essayez :

```
Claude, use paperqa_build_index to index my papers
```

Puis :

```
Claude, use paperqa_ask to answer: "What are the main contributions of the papers in my collection?"
```

## Exemples d'utilisation

### Poser une question spécifique

```
Claude, j'ai des papers sur le machine learning.
Utilise paperqa_ask pour me dire : "What are the latest techniques for improving neural network efficiency?"
```

### Ajouter un nouveau paper

```
Claude, ajoute ce paper à ma collection avec paperqa_add_paper :
/home/user/Downloads/new_paper.pdf
```

### Rechercher un sujet

```
Claude, utilise paperqa_search pour trouver tous les papers qui mentionnent "transformer architecture"
```

### Lister tous les documents

```
Claude, montre-moi tous les papers dans ma collection avec paperqa_list_docs
```

## Configuration avancée

### Utiliser des presets différents

Éditez `~/.config/claude/mcp_config.json` :

```json
{
  "mcpServers": {
    "paperqa": {
      "command": "paperqa-mcp",
      "env": {
        "PAPERQA_SETTINGS": "high_quality"
      }
    }
  }
}
```

Presets disponibles :
- `fast` - Rapide, moins précis (recommandé pour débuter)
- `high_quality` - Lent, très précis (meilleur qualité)
- `wikicrow` - Génération d'articles détaillés

### Utiliser Claude au lieu d'OpenAI

```json
{
  "mcpServers": {
    "paperqa": {
      "command": "paperqa-mcp",
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "PAPERQA_SETTINGS": "fast"
      }
    }
  }
}
```

Puis modifiez le preset ou créez un settings personnalisé qui utilise Claude.

## Dépannage rapide

### "No documents found"

→ Vérifiez que vous avez des PDFs dans le dossier papers :
```bash
ls ~/papers/*.pdf
```

### "OPENAI_API_KEY not set"

→ Ajoutez votre clé à votre shell :
```bash
echo 'export OPENAI_API_KEY="sk-..."' >> ~/.bashrc
source ~/.bashrc
```

Ou ajoutez-la directement dans `mcp_config.json`

### Le serveur ne démarre pas

→ Vérifiez l'installation :
```bash
which paperqa-mcp
paperqa-mcp --help
```

### Les réponses sont vides

→ Construisez l'index d'abord :
```
Claude, use paperqa_build_index
```

## Workflows recommandés

### Workflow 1 : Recherche sur un nouveau sujet

```
1. Claude, recherche "quantum computing" avec paperqa_search
2. Ensuite, utilise paperqa_ask pour répondre : "What are the key challenges in quantum error correction?"
3. Liste les sources avec paperqa_list_docs
```

### Workflow 2 : Ajouter et analyser un nouveau paper

```
1. Claude, ajoute ~/Downloads/paper.pdf avec paperqa_add_paper
2. Puis pose la question avec paperqa_ask : "What is the main contribution of this paper?"
```

### Workflow 3 : Analyse comparative

```
Claude, j'ai plusieurs papers sur les transformers.
Utilise paperqa_ask pour comparer : "How do the different papers approach attention mechanisms?"
```

## Ressources

- Documentation complète : voir `README.md`
- Issues et support : [GitHub Issues](https://github.com/Future-House/paper-qa/issues)
- Documentation PaperQA : https://github.com/Future-House/paper-qa
