# Fonctionnalités du PaperQA MCP Server

## Vue d'ensemble

Le serveur MCP PaperQA expose 6 outils puissants pour interagir avec des articles scientifiques directement depuis Claude Code.

## Outils disponibles

### 1. 🔍 `paperqa_ask` - Poser des questions

Posez des questions sur vos articles scientifiques et obtenez des réponses **citées** et **vérifiées**.

**Cas d'usage :**
- Comprendre un concept complexe à travers plusieurs papers
- Comparer les approches de différents auteurs
- Trouver les contributions principales d'un domaine
- Obtenir un résumé d'un sujet spécifique

**Exemple :**
```
Claude, use paperqa_ask to answer:
"What are the main advantages of transformer architectures over RNNs?"
```

**Sortie typique :**
- Réponse synthétisée
- Citations des sources
- Extraits pertinents des papers

---

### 2. 📄 `paperqa_add_paper` - Ajouter des documents

Ajoutez des articles à votre collection pour les interroger ultérieurement.

**Formats supportés :**
- PDF (le plus courant)
- Fichiers texte (.txt, .md)
- URLs (téléchargement automatique)
- HTML
- Documents Office (docx, pptx)

**Cas d'usage :**
- Construire une bibliothèque personnalisée
- Ajouter un nouveau paper juste publié
- Importer depuis arXiv ou d'autres sources

**Exemple :**
```
Claude, use paperqa_add_paper to add:
/home/user/Downloads/nature_paper.pdf
```

**Options :**
- Citation personnalisée
- Nom de document personnalisé
- Téléchargement depuis URL

---

### 3. 🔎 `paperqa_search` - Recherche par mots-clés

Effectuez une recherche **full-text** dans vos documents indexés.

**Cas d'usage :**
- Trouver tous les papers mentionnant un terme spécifique
- Identifier les documents pertinents avant de poser une question
- Explorer rapidement votre collection

**Exemple :**
```
Claude, use paperqa_search to find papers about "CRISPR gene editing"
```

**Sortie :**
- Liste de documents correspondants
- Extraits pertinents
- Chemins des fichiers

**Différence avec `paperqa_ask` :**
- `paperqa_search` : Recherche simple par mots-clés
- `paperqa_ask` : Analyse sémantique + génération de réponse

---

### 4. 🗂️ `paperqa_build_index` - Construire un index

Créez ou mettez à jour l'index de recherche pour accélérer les requêtes.

**Quand l'utiliser :**
- Après avoir ajouté plusieurs nouveaux papers
- Première utilisation du serveur
- Après avoir modifié le dossier de papers

**Cas d'usage :**
- Indexer un nouveau dossier de papers
- Reconstruire l'index après des modifications
- Créer plusieurs index pour différents projets

**Exemple :**
```
Claude, use paperqa_build_index to index all papers in ~/research-papers
```

**Notes :**
- L'indexation peut prendre quelques minutes pour de grandes collections
- L'index est persisté sur disque
- Améliore significativement la vitesse de recherche

---

### 5. 📚 `paperqa_list_docs` - Lister les documents

Affichez tous les documents actuellement dans votre collection.

**Cas d'usage :**
- Vérifier quels papers sont disponibles
- Obtenir les métadonnées des documents
- Valider que l'ajout de documents a réussi

**Exemple :**
```
Claude, use paperqa_list_docs to show all my papers
```

**Informations affichées :**
- Titre du document
- Auteurs
- Année de publication
- Clé unique du document

---

### 6. ⚙️ `paperqa_get_settings` - Obtenir la configuration

Affichez la configuration actuelle du serveur PaperQA.

**Cas d'usage :**
- Vérifier les paramètres actifs
- Déboguer des problèmes de configuration
- Confirmer les chemins de dossiers

**Exemple :**
```
Claude, use paperqa_get_settings to show the current configuration
```

**Informations affichées :**
- Dossier de papers
- Dossier d'index
- Type d'agent
- Modèle LLM utilisé
- Modèle d'embedding
- Paramètres de recherche

---

## Workflows recommandés

### Workflow 1 : Première utilisation

```
1. paperqa_build_index     → Indexer vos papers
2. paperqa_list_docs       → Vérifier les documents
3. paperqa_ask             → Poser votre première question
```

### Workflow 2 : Ajouter et analyser un nouveau paper

```
1. paperqa_add_paper       → Ajouter le nouveau paper
2. paperqa_ask             → Analyser son contenu
3. paperqa_search          → Trouver des papers similaires
```

### Workflow 3 : Recherche approfondie

```
1. paperqa_search          → Recherche initiale large
2. paperqa_ask             → Question spécifique
3. paperqa_ask             → Question de suivi
4. paperqa_list_docs       → Identifier les sources clés
```

### Workflow 4 : Gestion de collection

```
1. paperqa_list_docs       → État actuel
2. paperqa_add_paper (×N)  → Ajout de plusieurs papers
3. paperqa_build_index     → Reconstruire l'index
4. paperqa_get_settings    → Vérifier la config
```

---

## Comparaison des outils

| Outil | Vitesse | Précision | Cas d'usage principal |
|-------|---------|-----------|----------------------|
| `paperqa_ask` | Lent (10-30s) | Très haute | Questions complexes nécessitant analyse |
| `paperqa_search` | Rapide (2-5s) | Moyenne | Recherche rapide de documents |
| `paperqa_add_paper` | Moyenne (5-10s) | N/A | Gestion de collection |
| `paperqa_build_index` | Lent (variable) | N/A | Préparation/optimisation |
| `paperqa_list_docs` | Très rapide (<1s) | N/A | Consultation |
| `paperqa_get_settings` | Très rapide (<1s) | N/A | Configuration |

---

## Fonctionnalités avancées

### Recherche multi-critères

Combinez `paperqa_search` et `paperqa_ask` :

```
1. Recherchez "transformer AND attention"
2. Ensuite posez une question spécifique sur les résultats
```

### Citations et sources

`paperqa_ask` fournit automatiquement :
- Citations dans le texte
- Liste de sources
- Extraits pertinents des papers

### Support multimodal

PaperQA peut analyser :
- Texte des articles
- Figures et images (avec légendes)
- Tableaux de données

### Index multiples

Créez des index différents pour différents projets :

```
paperqa_build_index avec index_name="machine-learning"
paperqa_build_index avec index_name="quantum-computing"
```

---

## Limitations actuelles

1. **Langues** : Fonctionne mieux avec des papers en anglais
2. **Format** : Les PDFs scannés (images) nécessitent OCR
3. **Taille** : Très gros documents peuvent être tronqués
4. **Coût** : Utilise des APIs LLM (OpenAI par défaut)

---

## Prochaines fonctionnalités (roadmap)

- [ ] Support de l'analyse de graphiques et équations
- [ ] Export de réponses en formats structurés (JSON, BibTeX)
- [ ] Recherche par similarité sémantique
- [ ] Gestion de tags et catégories
- [ ] Support de bases de données vectorielles externes
- [ ] Interface web pour visualisation

---

## Ressources

- [Documentation complète](README.md)
- [Guide de démarrage rapide](QUICKSTART.md)
- [Guide de test](TESTING.md)
- [Documentation PaperQA](https://github.com/Future-House/paper-qa)
