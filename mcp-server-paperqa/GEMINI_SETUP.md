# Configuration PaperQA avec Gemini

Guide complet pour utiliser **Google Gemini** comme LLM principal, summary LLM, et embeddings dans PaperQA MCP Server.

## 🎯 Objectif

Utiliser uniquement Gemini pour :
- ✅ **LLM principal** : Génération des réponses
- ✅ **Summary LLM** : Résumés des extraits de papers
- ✅ **Embeddings** : Vectorisation des textes

## 📋 Prérequis

### 1. Obtenir une clé API Google

1. Allez sur : https://makersuite.google.com/app/apikey
2. Créez un projet (si nécessaire)
3. Cliquez sur "Create API Key"
4. Copiez votre clé (format : `AIzaSy...`)

### 2. Vérifier les quotas

Gemini offre des quotas généreux :
- **Gratuit** : 15 requêtes/minute, 1500/jour
- **Payant** : Quotas beaucoup plus élevés

Pour production, activez la facturation : https://console.cloud.google.com/billing

## 🚀 Installation rapide

### Étape 1 : Installer le serveur MCP

```bash
cd /home/user/paper-qa/mcp-server-paperqa
./install.sh
```

### Étape 2 : Configurer Claude Code avec Gemini

Copiez la configuration Gemini :

```bash
cp mcp_config.gemini.json ~/.config/claude/mcp_config.json
```

### Étape 3 : Éditer la configuration

Ouvrez `~/.config/claude/mcp_config.json` et modifiez :

```json
{
  "mcpServers": {
    "paperqa": {
      "command": "paperqa-mcp",
      "env": {
        "GOOGLE_API_KEY": "AIzaSy...",  ← VOTRE CLÉ ICI
        "PAPERQA_PAPER_DIRECTORY": "/home/user/papers",  ← VOTRE DOSSIER
        "PAPERQA_INDEX_DIRECTORY": "/home/user/.paperqa/indexes",
        "PAPERQA_SETTINGS": "/home/user/paper-qa/mcp-server-paperqa/configs/gemini-full.json"
      }
    }
  }
}
```

### Étape 4 : Créer le dossier papers

```bash
mkdir -p ~/papers
```

### Étape 5 : Redémarrer Claude Code

Fermez et relancez Claude Code.

## ⚙️ Configurations disponibles

J'ai créé **3 presets** optimisés pour Gemini :

### 1. `gemini-fast.json` - Rapide et économique ⚡

**Utilise** : Gemini 2.0 Flash (le plus récent)

**Avantages** :
- Très rapide (~5-10 secondes par question)
- Très économique
- Bon pour exploration rapide

**Configuration** :
```json
{
  "llm": "gemini/gemini-2.0-flash-exp",
  "summary_llm": "gemini/gemini-2.0-flash-exp",
  "embedding": "text-embedding-004",
  "answer": {
    "evidence_k": 5,
    "max_sources": 3
  }
}
```

**Utiliser** :
```json
"PAPERQA_SETTINGS": "/home/user/paper-qa/mcp-server-paperqa/configs/gemini-fast.json"
```

---

### 2. `gemini-full.json` - Équilibré (RECOMMANDÉ) ⭐

**Utilise** : Gemini 1.5 Flash

**Avantages** :
- Bon équilibre qualité/vitesse/coût
- Réponses détaillées
- Fiable et testé

**Configuration** :
```json
{
  "llm": "gemini/gemini-1.5-flash",
  "summary_llm": "gemini/gemini-1.5-flash",
  "embedding": "text-embedding-004",
  "answer": {
    "evidence_k": 10,
    "max_sources": 5
  }
}
```

**Utiliser** :
```json
"PAPERQA_SETTINGS": "/home/user/paper-qa/mcp-server-paperqa/configs/gemini-full.json"
```

---

### 3. `gemini-high-quality.json` - Meilleure qualité 🏆

**Utilise** : Gemini 1.5 Pro (LLM) + Gemini 1.5 Flash (summary)

**Avantages** :
- Meilleure qualité de réponse
- Plus d'evidence et de sources
- Réponses plus détaillées

**Configuration** :
```json
{
  "llm": "gemini/gemini-1.5-pro",
  "summary_llm": "gemini/gemini-1.5-flash",
  "embedding": "text-embedding-004",
  "answer": {
    "evidence_k": 15,
    "max_sources": 8
  }
}
```

**Utiliser** :
```json
"PAPERQA_SETTINGS": "/home/user/paper-qa/mcp-server-paperqa/configs/gemini-high-quality.json"
```

---

## 📊 Comparaison des presets

| Preset | Vitesse | Qualité | Coût | Usage recommandé |
|--------|---------|---------|------|------------------|
| **gemini-fast** | ⚡⚡⚡ | ⭐⭐ | 💰 | Tests, exploration rapide |
| **gemini-full** | ⚡⚡ | ⭐⭐⭐ | 💰💰 | Usage quotidien |
| **gemini-high-quality** | ⚡ | ⭐⭐⭐⭐ | 💰💰💰 | Recherche approfondie |

## 🧪 Test de la configuration

### Test 1 : Vérifier que le serveur démarre

Dans Claude Code :

```
Claude, what MCP servers are available?
```

Devrait lister **"paperqa"**.

### Test 2 : Vérifier la configuration

```
Claude, use paperqa_get_settings
```

Devrait afficher :
- LLM Model: `gemini/gemini-1.5-flash` (ou autre selon votre preset)
- Embedding Model: `text-embedding-004`
- Paper Directory: votre dossier

### Test 3 : Ajouter un paper de test

```bash
# Télécharger un paper d'exemple
cd ~/papers
wget https://arxiv.org/pdf/1706.03762.pdf -O transformer.pdf
```

Dans Claude Code :

```
Claude, use paperqa_build_index to index my papers
```

### Test 4 : Poser une question

```
Claude, use paperqa_ask to answer: "What is the transformer architecture?"
```

Devrait retourner une réponse citée avec sources.

## 🔧 Configuration avancée

### Modifier les paramètres

Vous pouvez éditer les fichiers JSON dans `configs/` pour ajuster :

**Nombre de sources** :
```json
{
  "answer": {
    "max_sources": 5  ← Augmenter pour plus de citations
  }
}
```

**Longueur de réponse** :
```json
{
  "answer": {
    "answer_length": "about 300 words"  ← Plus long/court
  }
}
```

**Nombre d'evidence** :
```json
{
  "answer": {
    "evidence_k": 10  ← Plus = meilleure qualité mais plus lent
  }
}
```

### Modèles Gemini disponibles

| Modèle | Description | Coût |
|--------|-------------|------|
| `gemini/gemini-2.0-flash-exp` | Plus récent, très rapide (expérimental) | $ |
| `gemini/gemini-1.5-flash` | Rapide, fiable, bon rapport qualité/prix | $ |
| `gemini/gemini-1.5-pro` | Meilleure qualité, plus lent | $$$ |
| `gemini/gemini-1.0-pro` | Ancien, stable | $ |

### Embeddings Gemini

Pour les embeddings, utilisez **toujours** :
```json
{
  "embedding": "text-embedding-004"
}
```

C'est le modèle d'embedding le plus récent de Google (janvier 2024).

**Dimensions** : 768
**Performances** : Comparables à `text-embedding-3-small` d'OpenAI

## 🐛 Dépannage

### Erreur "GOOGLE_API_KEY not set"

**Solution** :
```bash
export GOOGLE_API_KEY="AIzaSy..."
# Ajoutez à ~/.bashrc pour le rendre permanent
echo 'export GOOGLE_API_KEY="AIzaSy..."' >> ~/.bashrc
```

Ou ajoutez-le dans `mcp_config.json`.

### Erreur "429 Too Many Requests"

**Cause** : Quota Gemini dépassé

**Solutions** :
1. Attendez quelques minutes
2. Activez la facturation pour des quotas plus élevés
3. Utilisez un preset "fast" (moins de requêtes)

### Erreur "text-embedding-004 not found"

**Solution** : Vérifiez que votre clé API a accès à l'API Embeddings :
https://ai.google.dev/gemini-api/docs/embeddings

Si problème, utilisez OpenAI pour embeddings :
```json
{
  "embedding": "text-embedding-3-small",
  "env": {
    "GOOGLE_API_KEY": "...",
    "OPENAI_API_KEY": "sk-..."
  }
}
```

### Le serveur est lent

**Causes possibles** :
1. Utilisation de `gemini-1.5-pro` (plus lent)
2. Trop d'evidence (`evidence_k` élevé)
3. Gros documents

**Solutions** :
- Utilisez `gemini-fast.json`
- Réduisez `evidence_k` et `max_sources`
- Filtrez vos papers avant indexation

### Réponses de mauvaise qualité

**Solutions** :
1. Utilisez `gemini-high-quality.json`
2. Augmentez `evidence_k` à 15-20
3. Vérifiez que vos papers sont bien indexés :
   ```
   Claude, use paperqa_list_docs
   ```

## 💰 Coûts estimés

Avec Gemini (tarifs approximatifs 2024) :

**gemini-fast** :
- ~0.01$ par question
- ~1$ pour 100 questions

**gemini-full** :
- ~0.02$ par question
- ~2$ pour 100 questions

**gemini-high-quality** :
- ~0.05$ par question
- ~5$ pour 100 questions

**Gratuit** : ~100-200 questions/jour selon le preset

## 🎓 Exemples d'utilisation

### Exemple 1 : Recherche sur un sujet

```
1. Claude, use paperqa_search to find "quantum computing"
2. Claude, use paperqa_ask: "What are the main challenges in quantum error correction?"
```

### Exemple 2 : Analyser un nouveau paper

```
1. Téléchargez le paper dans ~/papers
2. Claude, use paperqa_build_index
3. Claude, use paperqa_ask: "What is the main contribution of the latest paper?"
```

### Exemple 3 : Comparaison de méthodes

```
Claude, use paperqa_ask: "Compare the transformer architecture and RNN approaches for NLP tasks"
```

## 📚 Ressources

- **API Gemini** : https://ai.google.dev/
- **Tarifs Gemini** : https://ai.google.dev/pricing
- **LiteLLM Docs** : https://docs.litellm.ai/docs/providers/gemini
- **PaperQA Docs** : https://github.com/Future-House/paper-qa

## ✅ Checklist de configuration

- [ ] Clé API Google obtenue
- [ ] `mcp_config.json` configuré avec votre clé
- [ ] Dossier papers créé et rempli
- [ ] Claude Code redémarré
- [ ] Test `paperqa_get_settings` réussi
- [ ] Index construit avec `paperqa_build_index`
- [ ] Première question testée avec `paperqa_ask`

---

**Vous êtes maintenant prêt à utiliser PaperQA avec Gemini ! 🚀**
