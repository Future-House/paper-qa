# Guide de test - PaperQA MCP Server

Ce document décrit comment tester le serveur MCP PaperQA.

## Tests unitaires

### Installation des dépendances de développement

```bash
pip install -e ".[dev]"
```

Ou :

```bash
pip install -r requirements.txt
```

### Lancer les tests

```bash
pytest tests/ -v
```

### Tests spécifiques

```bash
# Test de listing des outils
pytest tests/test_server.py::test_list_tools -v

# Test des schémas
pytest tests/test_server.py::test_paperqa_ask_tool_schema -v
```

## Test manuel du serveur

### 1. Vérifier l'installation

```bash
which paperqa-mcp
```

Devrait afficher le chemin vers l'exécutable.

### 2. Test de démarrage (mode debug)

Le serveur MCP utilise stdio pour la communication. Pour tester en mode interactif :

```bash
echo '{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}' | paperqa-mcp
```

Note : Le serveur attend des messages JSON-RPC via stdin/stdout.

### 3. Test avec un client MCP

Le meilleur moyen de tester est d'utiliser Claude Code directement :

1. Configurez le serveur dans `~/.config/claude/mcp_config.json`
2. Redémarrez Claude Code
3. Demandez à Claude : "What MCP servers are available?"
4. Claude devrait lister "paperqa"

## Tests d'intégration avec Claude Code

### Test 1 : Lister les outils

Dans Claude Code :

```
Claude, what tools are available from the paperqa server?
```

Devrait lister : `paperqa_ask`, `paperqa_add_paper`, `paperqa_search`, etc.

### Test 2 : Obtenir la configuration

```
Claude, use paperqa_get_settings to show me the current configuration
```

Devrait afficher :
- Paper directory
- Index directory
- Agent type
- LLM model
- etc.

### Test 3 : Lister les documents

```
Claude, use paperqa_list_docs to show all documents
```

Devrait lister les documents (ou indiquer que la collection est vide).

### Test 4 : Construire un index

Créez d'abord un dossier de test avec un PDF :

```bash
mkdir -p ~/test-papers
cp some-paper.pdf ~/test-papers/
```

Puis dans Claude Code :

```
Claude, use paperqa_build_index to index papers in ~/test-papers
```

### Test 5 : Poser une question

```
Claude, use paperqa_ask to answer: "What is machine learning?"
```

Si vous avez des papers pertinents, vous devriez obtenir une réponse citée.

## Dépannage des tests

### pytest ne trouve pas les modules

```bash
pip install -e .
```

### Erreur "mcp module not found"

```bash
pip install mcp
```

### Erreur "paper-qa module not found"

```bash
cd /home/user/paper-qa
pip install -e .
```

### Le serveur ne démarre pas dans Claude Code

1. Vérifiez les logs de Claude Code
2. Sur macOS : `~/Library/Logs/Claude/`
3. Sur Linux : `~/.config/claude/logs/`

### Erreur OPENAI_API_KEY

Assurez-vous que votre clé API est définie :

```bash
export OPENAI_API_KEY="sk-..."
```

Ou ajoutez-la dans `mcp_config.json`.

## Validation complète

Checklist pour vérifier que tout fonctionne :

- [ ] `pytest tests/` passe tous les tests
- [ ] `paperqa-mcp` est dans le PATH
- [ ] Le serveur est listé dans Claude Code
- [ ] `paperqa_get_settings` fonctionne
- [ ] `paperqa_list_docs` fonctionne
- [ ] `paperqa_build_index` peut créer un index
- [ ] `paperqa_search` peut rechercher
- [ ] `paperqa_ask` peut répondre à une question simple

## Performance

### Benchmarks basiques

Pour mesurer le temps de réponse :

```bash
time paperqa-mcp < test_request.json
```

Temps attendus (avec settings "fast") :
- `paperqa_get_settings` : < 1s
- `paperqa_list_docs` : < 1s
- `paperqa_search` : 2-5s
- `paperqa_ask` : 10-30s (selon la complexité)
- `paperqa_build_index` : Variable selon le nombre de papers

## Contribution

Pour contribuer des tests :

1. Ajoutez vos tests dans `tests/`
2. Utilisez `pytest` et `pytest-asyncio`
3. Moquez les appels API avec `pytest-mock` ou VCR
4. Assurez-vous que les tests passent :

```bash
pytest tests/ -v
ruff check src/
black --check src/
```
