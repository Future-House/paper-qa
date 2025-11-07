# Configuration PaperQA avec Gemini 2.x

Guide pour utiliser les **derniers modèles Gemini** (Gemini 2.5 Pro, 2.0 Flash, 2.0 Flash Thinking) avec PaperQA.

## 🚀 Modèles Gemini 2.x disponibles

### Gemini 2.5 Pro - Le plus puissant 🏆

**Capacités** :
- Raisonnement avancé et analyse profonde
- Contexte ultra-long (jusqu'à 2M tokens)
- Meilleure compréhension multimodale
- Performances état de l'art

**Usage recommandé** :
- Recherche scientifique approfondie
- Analyse de nombreux papers complexes
- Questions nécessitant raisonnement multi-étapes
- Synthèses détaillées

**Configuration** : `gemini-2-5-pro.json`

---

### Gemini 2.0 Flash Thinking - Raisonnement explicite 🧠

**Capacités** :
- Mode "thinking" avec raisonnement visible
- Excellent pour questions complexes
- Rapide malgré le mode thinking
- Bon rapport qualité/prix

**Usage recommandé** :
- Questions complexes nécessitant analyse
- Comparaisons méthodologiques
- Problèmes multi-étapes
- Quand vous voulez comprendre le raisonnement

**Configuration** : `gemini-2-thinking.json`

---

### Gemini 2.0 Flash - Rapide et efficace ⚡

**Capacités** :
- Très rapide
- Économique
- Bonne qualité générale
- Multimodal natif

**Usage recommandé** :
- Usage quotidien
- Questions simples à moyennes
- Exploration rapide
- Tests

**Configuration** : `gemini-2-flash.json`

---

## 📊 Comparaison détaillée

| Modèle | Vitesse | Qualité | Contexte | Coût | Meilleur pour |
|--------|---------|---------|----------|------|---------------|
| **Gemini 2.5 Pro** | ⚡ | ⭐⭐⭐⭐⭐ | 2M tokens | $$$$ | Recherche approfondie |
| **Gemini 2.0 Flash Thinking** | ⚡⚡ | ⭐⭐⭐⭐ | 1M tokens | $$ | Raisonnement complexe |
| **Gemini 2.0 Flash** | ⚡⚡⚡ | ⭐⭐⭐ | 1M tokens | $ | Usage quotidien |
| *Gemini 1.5 Pro* | ⚡ | ⭐⭐⭐⭐ | 2M tokens | $$$ | Alternative stable |
| *Gemini 1.5 Flash* | ⚡⚡ | ⭐⭐⭐ | 1M tokens | $ | Fiable et testé |

---

## 🔧 Installation avec Gemini 2.5 Pro

### Étape 1 : Clé API

Obtenez votre clé API sur : https://makersuite.google.com/app/apikey

### Étape 2 : Configuration

Copiez la configuration pour Gemini 2.5 Pro :

```bash
cd /home/user/paper-qa/mcp-server-paperqa
cp mcp_config.gemini-2-5-pro.json ~/.config/claude/mcp_config.json
```

### Étape 3 : Éditer la configuration

Ouvrez `~/.config/claude/mcp_config.json` :

```json
{
  "mcpServers": {
    "paperqa": {
      "command": "paperqa-mcp",
      "env": {
        "GOOGLE_API_KEY": "AIzaSy...",  ← VOTRE CLÉ
        "PAPERQA_PAPER_DIRECTORY": "/home/user/papers",
        "PAPERQA_INDEX_DIRECTORY": "/home/user/.paperqa/indexes",
        "PAPERQA_SETTINGS": "/home/user/paper-qa/mcp-server-paperqa/configs/gemini-2-5-pro.json"
      }
    }
  }
}
```

### Étape 4 : Redémarrer Claude Code

---

## ⚙️ Configurations disponibles

### 1. `gemini-2-5-pro.json` - Maximum performance 🏆

**Modèles** :
- LLM : `gemini/gemini-2.5-pro`
- Summary : `gemini/gemini-2.0-flash-exp`
- Embedding : `text-embedding-004`

**Paramètres** :
- Evidence : 20 extraits
- Sources : 10 max
- Réponse : ~400 mots
- Search : 15 résultats

**Coût** : ~$0.10-0.15 par question
**Vitesse** : 30-60 secondes
**Qualité** : ⭐⭐⭐⭐⭐

**Utiliser** :
```json
"PAPERQA_SETTINGS": "/home/user/paper-qa/mcp-server-paperqa/configs/gemini-2-5-pro.json"
```

---

### 2. `gemini-2-thinking.json` - Raisonnement explicite 🧠

**Modèles** :
- LLM : `gemini/gemini-2.0-flash-thinking-exp`
- Summary : `gemini/gemini-2.0-flash-exp`
- Embedding : `text-embedding-004`

**Paramètres** :
- Evidence : 12 extraits
- Sources : 6 max
- Réponse : ~250 mots
- Search : 10 résultats

**Coût** : ~$0.03-0.05 par question
**Vitesse** : 15-30 secondes
**Qualité** : ⭐⭐⭐⭐

**Utiliser** :
```json
"PAPERQA_SETTINGS": "/home/user/paper-qa/mcp-server-paperqa/configs/gemini-2-thinking.json"
```

---

### 3. `gemini-2-flash.json` - Rapide et efficace ⚡

**Modèles** :
- LLM : `gemini/gemini-2.0-flash-exp`
- Summary : `gemini/gemini-2.0-flash-exp`
- Embedding : `text-embedding-004`

**Paramètres** :
- Evidence : 10 extraits
- Sources : 5 max
- Réponse : ~200 mots
- Search : 8 résultats

**Coût** : ~$0.01-0.02 par question
**Vitesse** : 5-15 secondes
**Qualité** : ⭐⭐⭐

**Utiliser** :
```json
"PAPERQA_SETTINGS": "/home/user/paper-qa/mcp-server-paperqa/configs/gemini-2-flash.json"
```

---

## 🎯 Quel modèle choisir ?

### Pour vous (Gemini 2.5 Pro) : ✅ `gemini-2-5-pro.json`

**Pourquoi ?**
- Accès aux capacités les plus avancées
- Meilleure compréhension des papers complexes
- Réponses les plus détaillées et précises
- Analyse approfondie sur plusieurs papers

**Idéal pour** :
- Meta-analyses scientifiques
- Comparaisons de méthodes
- Synthèses de littérature
- Questions nécessitant raisonnement profond

### Si vous voulez économiser : `gemini-2-flash.json`

**Pourquoi ?**
- ~10x moins cher
- Toujours très bon
- Plus rapide
- Suffisant pour 80% des questions

### Si vous voulez voir le raisonnement : `gemini-2-thinking.json`

**Pourquoi ?**
- Mode thinking explicite
- Comprendre comment l'IA analyse
- Déboguer des réponses
- Questions complexes

---

## 🧪 Exemples d'utilisation

### Avec Gemini 2.5 Pro

**Question complexe** :
```
Claude, use paperqa_ask: "Compare the methodological approaches across all papers
on protein folding in my collection. What are the key differences and which
approach shows the most promise based on empirical results?"
```

**Meta-analyse** :
```
Claude, use paperqa_ask: "Synthesize the findings from all papers about CRISPR
gene editing safety. What are the consensus points and where do researchers
disagree? Provide specific citations for each claim."
```

**Analyse temporelle** :
```
Claude, use paperqa_ask: "How have transformer architecture designs evolved
from 2017 to 2024 based on my collection? Identify key innovations and their
impact on performance."
```

### Avec Gemini 2.0 Flash Thinking

**Raisonnement multi-étapes** :
```
Claude, use paperqa_ask: "If I want to implement a new neural architecture
combining ideas from these papers, what are the key design choices I need to
make and what are the tradeoffs?"
```

### Avec Gemini 2.0 Flash

**Questions rapides** :
```
Claude, use paperqa_ask: "What is the main contribution of the AlphaFold paper?"
```

---

## 💡 Conseils d'optimisation

### Pour Gemini 2.5 Pro

**Maximiser la qualité** :
```json
{
  "answer": {
    "evidence_k": 25,        // Plus d'evidence
    "max_sources": 15        // Plus de sources
  }
}
```

**Questions longues** :
- Gemini 2.5 Pro gère 2M tokens
- Vous pouvez poser des questions très détaillées
- Demander des analyses exhaustives

**Multimodal** :
- Peut analyser figures et tableaux
- Mieux que les versions précédentes

### Pour économiser

**Mode économique** :
```json
{
  "answer": {
    "evidence_k": 5,         // Moins d'evidence
    "max_sources": 3         // Moins de sources
  }
}
```

---

## 💰 Coûts estimés (2024-2025)

| Modèle | Input (1M tokens) | Output (1M tokens) | Coût/question moyen |
|--------|-------------------|-------------------|---------------------|
| Gemini 2.5 Pro | $2.50 | $10.00 | $0.10-0.15 |
| Gemini 2.0 Flash Thinking | $0.15 | $0.60 | $0.03-0.05 |
| Gemini 2.0 Flash | $0.10 | $0.40 | $0.01-0.02 |
| *Gemini 1.5 Pro* | $1.25 | $5.00 | $0.05-0.08 |
| *Gemini 1.5 Flash* | $0.075 | $0.30 | $0.01-0.02 |

*Prix indicatifs, vérifiez sur https://ai.google.dev/pricing*

---

## 🐛 Dépannage

### "Model gemini-2.5-pro not found"

**Cause** : Le modèle n'est peut-être pas encore disponible dans votre région ou nécessite un accès spécial.

**Solution** :
1. Vérifiez sur https://ai.google.dev/models/gemini
2. Utilisez `gemini-2.0-flash-exp` en attendant :
   ```json
   "llm": "gemini/gemini-2.0-flash-exp"
   ```

### "Quota exceeded"

**Cause** : Limites de taux dépassées

**Solutions** :
- Attendez quelques minutes
- Activez la facturation pour quotas plus élevés
- Utilisez un modèle moins sollicité (Flash)

### Réponses trop longues/courtes

**Ajuster la longueur** :
```json
{
  "answer": {
    "answer_length": "about 500 words"  // ou "100 words", etc.
  }
}
```

---

## 🔄 Migration depuis anciens modèles

### De Gemini 1.5 Pro → 2.5 Pro

Changez simplement dans votre config :
```json
{
  "llm": "gemini/gemini-2.5-pro"  // au lieu de gemini-1.5-pro
}
```

**Avantages** :
- Meilleure qualité (+20-30%)
- Contexte maintenu (2M tokens)
- Multimodal amélioré

**Coût** :
- 2x plus cher mais qualité supérieure

### De OpenAI → Gemini 2.x

**Équivalences** :
- GPT-4 Turbo → Gemini 2.5 Pro
- GPT-4 → Gemini 2.0 Flash Thinking
- GPT-3.5 Turbo → Gemini 2.0 Flash

**Avantages de Gemini** :
- Moins cher (2-5x)
- Contexte plus long
- Multimodal natif
- Quotas plus généreux

---

## 📚 Ressources

- **Gemini API** : https://ai.google.dev/
- **Modèles Gemini** : https://ai.google.dev/models/gemini
- **Tarifs** : https://ai.google.dev/pricing
- **Documentation** : https://ai.google.dev/docs

---

## ✅ Checklist rapide

- [ ] Clé API Google obtenue
- [ ] Configuration Gemini 2.5 Pro copiée
- [ ] `mcp_config.json` édité avec clé et chemins
- [ ] Dossier papers créé avec quelques PDFs
- [ ] Claude Code redémarré
- [ ] Test `paperqa_get_settings` réussi
- [ ] Index construit avec `paperqa_build_index`
- [ ] Première question testée

---

**Vous êtes maintenant prêt à utiliser PaperQA avec Gemini 2.5 Pro ! 🚀**

Pour questions complexes nécessitant analyse approfondie, c'est le meilleur choix disponible.
