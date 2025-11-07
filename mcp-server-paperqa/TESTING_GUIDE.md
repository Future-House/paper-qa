# Test de PaperQA - Guide rapide

## ✅ Tests réussis (sans API)

Tous les tests de base ont réussi :
- PaperQA installé et fonctionnel
- 6 configurations Gemini valides
- CLI disponible (`pqa` command)
- Structure complète prête

## 🧪 Pour tester avec une vraie API

### Option 1 : Gemini (GRATUIT - Recommandé)

**1. Obtenir une clé API Gemini :**
- Allez sur : https://makersuite.google.com/app/apikey
- Cliquez sur "Create API Key"
- Copiez votre clé (format : `AIzaSy...`)

**2. Configurer la clé :**
```bash
export GOOGLE_API_KEY="votre-clé-ici"
```

**3. Test rapide avec fichier texte :**
```bash
# Créer un dossier de test
mkdir -p ~/test-papers

# Créer un paper de test
cat > ~/test-papers/ml-basics.txt << 'EOF'
Machine Learning Basics

Machine learning is a subset of artificial intelligence. It enables
computers to learn from data without explicit programming. There are
three main types:
- Supervised learning
- Unsupervised learning
- Reinforcement learning

Applications include image recognition, natural language processing,
and autonomous vehicles.
EOF

# Poser une question avec Gemini 2.0 Flash
pqa ask "What is machine learning?" \
  --paper-directory ~/test-papers \
  --llm gemini/gemini-2.0-flash-exp \
  --summary-llm gemini/gemini-2.0-flash-exp \
  --embedding text-embedding-004
```

**4. Test avec un vrai PDF :**
```bash
# Télécharger un paper d'exemple
cd ~/test-papers
wget https://arxiv.org/pdf/1706.03762.pdf -O transformer.pdf

# Construire l'index
pqa index ~/test-papers \
  --llm gemini/gemini-2.0-flash-exp \
  --embedding text-embedding-004

# Poser une question
pqa ask "What is the transformer architecture?" \
  --llm gemini/gemini-2.0-flash-exp
```

**5. Utiliser une configuration pré-définie :**
```bash
# Copier une config Gemini
cp /home/user/paper-qa/mcp-server-paperqa/configs/gemini-2-flash.json ~/my-config.json

# Éditer pour pointer vers vos papers
# (modifier "paper_directory" dans le JSON)

# Utiliser la config
pqa ask "your question" \
  --settings ~/my-config.json \
  --llm gemini/gemini-2.0-flash-exp
```

### Option 2 : Avec OpenAI (si vous avez des crédits)

```bash
export OPENAI_API_KEY="sk-..."
pqa ask "What is machine learning?" \
  --paper-directory ~/test-papers
```

### Option 3 : Modèles locaux (100% gratuit, pas d'API)

```bash
# Installer Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Télécharger un modèle
ollama pull llama3.1
ollama pull nomic-embed-text

# Tester
pqa ask "What is machine learning?" \
  --paper-directory ~/test-papers \
  --llm ollama/llama3.1 \
  --summary-llm ollama/llama3.1 \
  --embedding ollama/nomic-embed-text
```

## 🔧 Dépannage

### Erreur "GOOGLE_API_KEY not set"
```bash
echo $GOOGLE_API_KEY  # Vérifier qu'elle est définie
export GOOGLE_API_KEY="AIzaSy..."  # La redéfinir si vide
```

### Erreur "text-embedding-004 not found"
Utilisez OpenAI embeddings à la place :
```bash
export OPENAI_API_KEY="sk-..."
pqa ask "question" \
  --llm gemini/gemini-2.0-flash-exp \
  --embedding text-embedding-3-small
```

### Le paper n'est pas trouvé
Vérifiez le dossier :
```bash
ls ~/test-papers/
pqa view --paper-directory ~/test-papers
```

## 📊 Commandes utiles

### Voir la configuration actuelle
```bash
pqa view --settings gemini-2-flash
```

### Construire un index
```bash
pqa index ~/papers --llm gemini/gemini-2.0-flash-exp
```

### Recherche simple (keyword)
```bash
pqa search "machine learning" --index default
```

### Sauvegarder une config personnalisée
```bash
pqa save my-custom-config --llm gemini/gemini-2.5-pro
```

## 🎯 Workflow recommandé

1. **Obtenir clé Gemini** (gratuit)
2. **Créer dossier papers** : `mkdir ~/papers`
3. **Ajouter quelques PDFs** dans `~/papers`
4. **Tester** : `pqa ask "test question" --llm gemini/gemini-2.0-flash-exp`
5. **Si ça marche → Passer au serveur MCP**

## 📝 Notes importantes

- **Gemini gratuit** : 1500 requêtes/jour - largement suffisant
- **Embeddings** : `text-embedding-004` pour Gemini
- **Config files** : Dans `/home/user/paper-qa/mcp-server-paperqa/configs/`
- **Logs** : Utilisez `--verbosity 1` pour plus de détails

## ✅ Une fois que PaperQA marche...

Passez au serveur MCP pour l'intégrer dans Claude Code :
```bash
cd /home/user/paper-qa/mcp-server-paperqa
pip install -e .
# Puis configurez mcp_config.json
```

Mais testez d'abord PaperQA directement avec `pqa` !
