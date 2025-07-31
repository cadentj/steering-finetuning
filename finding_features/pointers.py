sports = """- Look for features that consistently activate on people's names, particularly athletes or sports figures.
- Look for features that detect sports-related terminology, team names, competitions, or game activities."""

sentiment = """- Look for features that track words with strong positive or negative emotional connotations.
- Look for features that detect language patterns commonly used to express or describe feelings and reactions."""

verbs = """- Look for features that specifically recognize verbs across different tenses and forms.
- Look for features that recognize singular or plural noun forms.
- Look for features that activate on descriptions of grammatical concepts."""

pronouns = """- Look for features that primarily activate on personal pronouns (he, she, they, etc.).
- Look for features that recognize gendered terminology, names, or role descriptions."""

gender = """- Look for features that primarily activate on personal pronouns (he, she, they, etc.).
- Look for features that detect references to gender identity or gender-related concepts.
- Look for features that recognize gendered terminology, names, or role descriptions.
- Look for features that bias certain occupations or roles to one gender."""

pointers = {
   "sports": sports,
   "verbs": verbs,
   "sentiment": sentiment,
   "pronouns": pronouns,
   "gender": gender,
}