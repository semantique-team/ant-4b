"""Synthetic customer-triage benchmark: department routing with controlled
difficulty, hard negatives, paraphrases, and five languages.

Candidates are always English department labels (per the multilingual policy:
candidates are never translated automatically). States and questions are
localized. Gold labels follow the strongest signal in the text, including the
hard split where a confusable department is mentioned but is not the owner.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from ant.training.data import DecisionDataset
from ant.types import DecisionExample

DEPARTMENTS: tuple[str, ...] = (
    "Billing",
    "Technical Support",
    "Sales",
    "Legal",
    "Customer Success",
    "Security",
)

HARD_NEGATIVES: dict[str, tuple[str, ...]] = {
    "Billing": ("Sales", "Customer Success"),
    "Technical Support": ("Customer Success", "Billing"),
    "Sales": ("Billing", "Customer Success"),
    "Legal": ("Security",),
    "Customer Success": ("Sales", "Technical Support", "Billing"),
    "Security": ("Legal",),
}

LANGUAGES: tuple[str, ...] = ("en", "it", "es", "fr", "de")
DIFFICULTIES: tuple[str, ...] = ("easy", "medium", "hard")

_EN: dict[str, dict[str, list[str]]] = {
    "Billing": {
        "easy": [
            "The customer was charged twice for the same invoice and wants a refund.",
            "My card was billed three times for one month of the subscription.",
            "The invoice amount is wrong, it does not match the order total.",
            "I keep getting charged after I cancelled my plan.",
        ],
        "medium": [
            "The payment went through twice when I renewed, and the bank confirmed both transactions.",
            "There is a duplicate charge on my statement from last week for the same order.",
            "I was overcharged on my annual plan and I have asked for the difference back.",
        ],
        "hard": [
            "The sales representative promised a 20% discount, but the invoice I received shows the full price.",
            "I already spoke with someone from the sales team about upgrading, and now my card shows two charges for the same month.",
            "Customer success scheduled a renewal call with me, but before that happened I was billed twice for April.",
        ],
    },
    "Technical Support": {
        "easy": [
            "The mobile app crashes every time I open the reports page.",
            "I cannot log in, the app says my session expired immediately after entering the password.",
            "Exports from the dashboard produce a corrupted file that will not open.",
            "The API returns a 500 error whenever I send a request with attachments.",
        ],
        "medium": [
            "Since the last update, the sync between the web app and my phone stops at 80% and never finishes.",
            "My webhook endpoint stopped receiving events three days ago and nothing was changed on our side.",
            "The search feature returns results in the wrong language and crashes on long queries.",
        ],
        "hard": [
            "The onboarding agent from customer success helped me set up the workspace, but now the integrations page fails to load.",
            "After the billing plan was upgraded, the SSO login started rejecting all of my team members.",
            "Our account manager in sales asked us to test the new module, and now the scheduler service will not start.",
        ],
    },
    "Sales": {
        "easy": [
            "I would like a demo of the enterprise plan for a team of forty people.",
            "What discounts are available if we purchase two hundred seats?",
            "We are comparing vendors and need a commercial quote before the end of the month.",
            "Can you tell me the price difference between the pro and the business tiers?",
        ],
        "medium": [
            "Our procurement team needs a volume-based offer for the next fiscal year.",
            "We are a startup evaluating your platform against two competitors and need licensing options.",
            "I need to understand which plan fits a company that expects to triple in size.",
        ],
        "hard": [
            "Billing says our current plan supports ten users, but we want to expand to fifty and need new pricing.",
            "The invoice shows we are on the business tier, but we were promised enterprise features during the renewal discussion.",
            "Technical support fixed our usage limits, and now we want to discuss an upgrade for next year.",
        ],
    },
    "Legal": {
        "easy": [
            "We received a subpoena requesting user data for a court case.",
            "Our legal department needs the current data processing agreement before signing the renewal.",
            "A lawyer sent a formal notice claiming our product infringes a patent.",
            "We need to review the terms of service for a regulated industry deployment.",
        ],
        "medium": [
            "A competitor claims our marketing materials misuse their trademark and is threatening action.",
            "Our compliance officer must audit the sub-processors listed in the privacy policy.",
            "The regulator asked for documentation about data retention periods.",
        ],
        "hard": [
            "The security team reported a breach, and we must notify the supervisory authority within 72 hours.",
            "Customer success forwarded a customer who refuses the updated privacy terms and cites GDPR rights.",
            "The double refund requested from billing also involves a contractual dispute about the SLA.",
        ],
    },
    "Customer Success": {
        "easy": [
            "Our onboarding has stalled and nobody has contacted us in two weeks.",
            "I want to schedule the quarterly business review that was promised in our plan.",
            "How do we get training sessions for the new team members we just hired?",
            "Our renewal is coming up and I want to go through the adoption report.",
        ],
        "medium": [
            "We are using only a fraction of the features we pay for and need an adoption plan.",
            "The kickoff workshop was never delivered after we signed the contract.",
            "Two of our departments need a structured rollout plan for the next semester.",
        ],
        "hard": [
            "The app keeps crashing during our onboarding workshops and the team cannot complete the training program.",
            "We were double billed last month, and while the refund is being processed we also want to revise our renewal goals.",
            "Support fixed the login issue, but our users still need a full re-onboarding plan for the workspace.",
        ],
    },
    "Security": {
        "easy": [
            "I received a phishing email that impersonates your company.",
            "Our SOC 2 audit requires evidence of multi-factor authentication enforcement.",
            "Someone is attempting to brute-force our admin account.",
            "We need to report a vulnerability we found in your login flow.",
        ],
        "medium": [
            "The audit log shows logins from countries where we have no employees.",
            "We need the incident response playbook and disclosure policy for our risk assessment.",
            "An intern shared a production API key in a public repository.",
        ],
        "hard": [
            "Legal needs the chain of custody documentation after the credential stuffing attempt we reported.",
            "There is an active data leak suspicion, and both our DPO and the legal team are asking for a joint response.",
            "The phishing campaign targeted our billing data, and we need forensic confirmation of what was exposed.",
        ],
    },
}

_IT: dict[str, dict[str, list[str]]] = {
    "Billing": {
        "easy": [
            "Il cliente è stato addebitato due volte per la stessa fattura e chiede un rimborso.",
            "La carta è stata caricata tre volte per un solo mese di abbonamento.",
        ],
        "medium": ["Il pagamento è passato due volte al rinnovo e la banca ha confermato entrambe le transazioni."],
        "hard": ["Il commerciale aveva promesso uno sconto del 20%, ma la fattura ricevuta mostra il prezzo pieno."],
    },
    "Technical Support": {
        "easy": [
            "L'app mobile si blocca ogni volta che apro la pagina dei report.",
            "Non riesco ad accedere: la sessione scade subito dopo aver inserito la password.",
        ],
        "medium": ["Da ultimo aggiornamento la sincronizzazione tra web e telefono si ferma all'80%."],
        "hard": ["Dopo l'aggiornamento del piano, il login SSO rifiuta tutti i membri del team."],
    },
    "Sales": {
        "easy": [
            "Vorrei una demo del piano enterprise per un team di quaranta persone.",
            "Quali sconti sono disponibili per duecento postazioni?",
        ],
        "medium": ["Il team acquisti necessita un'offerta basata sul volume per il prossimo anno."],
        "hard": ["La fattura dice piano business, ma al rinnovo ci erano state promesse le funzioni enterprise."],
    },
    "Legal": {
        "easy": [
            "Abbiamo ricevuto un subpoena che richiede dati utente per un processo.",
            "Il dipartimento legale necessita l'accordo di trattamento dati prima del rinnovo.",
        ],
        "medium": ["Un concorrente sostiene che i nostri materiali marketing usano il loro marchio."],
        "hard": ["Il team sicurezza ha segnalato una violazione e dobbiamo notificare l'autorità entro 72 ore."],
    },
    "Customer Success": {
        "easy": [
            "Il nostro onboarding è fermo e nessuno ci contatta da due settimane.",
            "Voglio fissare il quarterly business review promesso nel piano.",
        ],
        "medium": ["Usiamo solo una parte delle funzioni pagate e ci serve un piano di adozione."],
        "hard": ["L'app si blocca durante i workshop di onboarding e il team non completa la formazione."],
    },
    "Security": {
        "easy": [
            "Ho ricevuto un'email di phishing che si spaccia per la vostra azienda.",
            "L'audit SOC 2 richiede evidenze dell'applicazione dell'autenticazione a due fattori.",
        ],
        "medium": ["Nei log di audit ci sono accessi da paesi in cui non abbiamo dipendenti."],
        "hard": ["La campagna di phishing prendeva di mira i dati di fatturazione e serve conferma forense."],
    },
}

_ES: dict[str, dict[str, list[str]]] = {
    "Billing": {
        "easy": [
            "Al cliente se le cobró dos veces la misma factura y pide un reembolso.",
            "Mi tarjeta fue cargada tres veces por un solo mes de suscripción.",
        ],
        "medium": ["El pago se procesó dos veces al renovar y el banco confirmó ambas transacciones."],
        "hard": ["El comercial prometió un descuento del 20%, pero la factura muestra el precio completo."],
    },
    "Technical Support": {
        "easy": [
            "La aplicación móvil se bloquea cada vez que abro la página de informes.",
            "No puedo iniciar sesión: la sesión caduca justo después de introducir la contraseña.",
        ],
        "medium": ["Desde la última actualización, la sincronización entre la web y el móvil se queda al 80%."],
        "hard": ["Tras ampliar el plan, el inicio de sesión SSO rechaza a todos los miembros del equipo."],
    },
    "Sales": {
        "easy": [
            "Quiero una demo del plan enterprise para un equipo de cuarenta personas.",
            "¿Qué descuentos hay disponibles para doscientos puestos?",
        ],
        "medium": ["El equipo de compras necesita una oferta por volumen para el próximo año fiscal."],
        "hard": ["Facturación dice que nuestro plan admite diez usuarios, pero queremos pasar a cincuenta y necesitamos precios nuevos."],
    },
    "Legal": {
        "easy": [
            "Hemos recibido una orden judicial que solicita datos de usuarios.",
            "El departamento jurídico necesita el acuerdo de tratamiento de datos antes de renovar.",
        ],
        "medium": ["Un competidor afirma que nuestro material de marketing usa su marca."],
        "hard": ["El equipo de seguridad informó de una brecha y debemos notificar a la autoridad en 72 horas."],
    },
    "Customer Success": {
        "easy": [
            "Nuestra implantación está parada y nadie nos contacta desde hace dos semanas.",
            "Quiero programar la revisión trimestral de negocio prometida en el plan.",
        ],
        "medium": ["Usamos solo una parte de las funciones que pagamos y necesitamos un plan de adopción."],
        "hard": ["La app se bloquea durante los talleres de onboarding y el equipo no termina la formación."],
    },
    "Security": {
        "easy": [
            "Recibí un correo de phishing que suplanta a su empresa.",
            "La auditoría SOC 2 requiere evidencias de la aplicación de la autenticación multifactor.",
        ],
        "medium": ["En los registros de auditoría hay accesos desde países donde no tenemos empleados."],
        "hard": ["La campaña de phishing apuntaba a los datos de facturación y necesitamos confirmación forense."],
    },
}

_FR: dict[str, dict[str, list[str]]] = {
    "Billing": {
        "easy": [
            "Le client a été débité deux fois pour la même facture et demande un remboursement.",
            "Ma carte a été débitée trois fois pour un seul mois d'abonnement.",
        ],
        "medium": ["Le paiement est passé deux fois lors du renouvellement et la banque a confirmé les deux transactions."],
        "hard": ["Le commercial avait promis une remise de 20 %, mais la facture reçue affiche le prix complet."],
    },
    "Technical Support": {
        "easy": [
            "L'application mobile plante chaque fois que j'ouvre la page des rapports.",
            "Je ne peux pas me connecter : la session expire juste après avoir saisi le mot de passe.",
        ],
        "medium": ["Depuis la dernière mise à jour, la synchronisation entre le web et le téléphone s'arrête à 80 %."],
        "hard": ["Après la montée en gamme de l'offre, la connexion SSO rejette tous les membres de l'équipe."],
    },
    "Sales": {
        "easy": [
            "Je voudrais une démonstration de l'offre entreprise pour une équipe de quarante personnes.",
            "Quelles remises sont possibles pour deux cents licences ?",
        ],
        "medium": ["Le service achats a besoin d'une offre au volume pour le prochain exercice."],
        "hard": ["La facturation indique que notre offre couvre dix utilisateurs, mais nous voulons passer à cinquante et il nous faut une nouvelle tarification."],
    },
    "Legal": {
        "easy": [
            "Nous avons reçu une assignation demandant des données d'utilisateurs dans le cadre d'une affaire.",
            "Le service juridique a besoin de l'accord de traitement des données avant le renouvellement.",
        ],
        "medium": ["Un concurrent prétend que nos supports marketing utilisent sa marque."],
        "hard": ["L'équipe sécurité a signalé une violation et nous devons notifier l'autorité sous 72 heures."],
    },
    "Customer Success": {
        "easy": [
            "Notre onboarding est bloqué et personne ne nous contacte depuis deux semaines.",
            "Je souhaite planifier la revue trimestrielle promise dans notre offre.",
        ],
        "medium": ["Nous n'utilisons qu'une fraction des fonctionnalités payées et il nous faut un plan d'adoption."],
        "hard": ["L'application plante pendant les ateliers d'onboarding et l'équipe ne termine pas la formation."],
    },
    "Security": {
        "easy": [
            "J'ai reçu un e-mail de phishing qui se fait passer pour votre entreprise.",
            "L'audit SOC 2 exige des preuves de l'application de l'authentification multifacteur.",
        ],
        "medium": ["Les journaux d'audit montrent des connexions depuis des pays où nous n'avons aucun employé."],
        "hard": ["La campagne de phishing ciblait les données de facturation et nous exigeons une confirmation forensique."],
    },
}

_DE: dict[str, dict[str, list[str]]] = {
    "Billing": {
        "easy": [
            "Dem Kunden wurde dieselbe Rechnung zweimal abgebucht, und er fordert eine Erstattung.",
            "Meine Karte wurde für einen einzigen Abo-Monat dreimal belastet.",
        ],
        "medium": ["Bei der Verlängerung wurde die Zahlung zweimal ausgeführt, und die Bank hat beide Transaktionen bestätigt."],
        "hard": ["Der Vertriebsmitarbeiter hat einen Rabatt von 20 % versprochen, aber die Rechnung zeigt den vollen Preis."],
    },
    "Technical Support": {
        "easy": [
            "Die mobile App stürzt jedes Mal ab, wenn ich die Berichtsseite öffne.",
            "Ich kann mich nicht anmelden: Die Sitzung läuft direkt nach der Passworteingabe ab.",
        ],
        "medium": ["Seit dem letzten Update bleibt die Synchronisierung zwischen Web und Handy bei 80 % hängen."],
        "hard": ["Nach dem Upgrade des Tarifs lehnt der SSO-Login alle Teammitglieder ab."],
    },
    "Sales": {
        "easy": [
            "Ich hätte gern eine Demonstration des Enterprise-Tarifs für ein Team von vierzig Personen.",
            "Welche Rabatte gibt es für zweihundert Arbeitsplätze?",
        ],
        "medium": ["Der Einkauf braucht ein Volumenangebot für das nächste Geschäftsjahr."],
        "hard": ["Die Abrechnung sagt, unser Tarif umfasst zehn Nutzer, aber wir wollen auf fünfzig erweitern und brauchen neue Preise."],
    },
    "Legal": {
        "easy": [
            "Wir haben eine Vorladung erhalten, die Kundendaten für ein Gerichtsverfahren verlangt.",
            "Die Rechtsabteilung braucht die Datenverarbeitungsvereinbarung vor der Vertragsverlängerung.",
        ],
        "medium": ["Ein Wettbewerber behauptet, unser Marketingmaterial nutze seine Marke."],
        "hard": ["Das Sicherheitsteam hat eine Verletzung gemeldet, und wir müssen die Behörde binnen 72 Stunden informieren."],
    },
    "Customer Success": {
        "easy": [
            "Unser Onboarding stockt, und seit zwei Wochen meldet sich niemand.",
            "Ich möchte das zugesagte vierteljährliche Geschäftsgespräch terminieren.",
        ],
        "medium": ["Wir nutzen nur einen Bruchteil der bezahlten Funktionen und brauchen einen Adoptionsplan."],
        "hard": ["Die App stürzt bei den Onboarding-Workshops ab, und das Team schließt die Schulung nicht ab."],
    },
    "Security": {
        "easy": [
            "Ich habe eine Phishing-E-Mail erhalten, die sich als Ihr Unternehmen ausgibt.",
            "Die SOC-2-Prüfung verlangt Nachweise zur Durchsetzung der Mehrfaktor-Authentifizierung.",
        ],
        "medium": ["In den Audit-Logs gibt es Anmeldungen aus Ländern, in denen wir keine Mitarbeiter haben."],
        "hard": ["Die Phishing-Kampagne zielte auf Abrechnungsdaten, und wir brauchen eine forensische Bestätigung."],
    },
}

TEMPLATES: dict[str, dict[str, dict[str, list[str]]]] = {
    "en": _EN,
    "it": _IT,
    "es": _ES,
    "fr": _FR,
    "de": _DE,
}

QUESTIONS: dict[str, tuple[str, ...]] = {
    "en": (
        "Which department should handle this case?",
        "Who should take care of this request?",
        "Route this case to the right team.",
    ),
    "it": (
        "Quale dipartimento dovrebbe gestire questo caso?",
        "Chi dovrebbe occuparsi di questa richiesta?",
        "Instrada questo caso al team giusto.",
    ),
    "es": (
        "¿Qué departamento debería gestionar este caso?",
        "¿Quién debería encargarse de esta solicitud?",
        "Asigna este caso al equipo correcto.",
    ),
    "fr": (
        "Quel service devrait traiter ce cas ?",
        "Qui devrait s'occuper de cette demande ?",
        "Aiguillez ce cas vers la bonne équipe.",
    ),
    "de": (
        "Welche Abteilung sollte diesen Fall bearbeiten?",
        "Wer sollte diese Anfrage übernehmen?",
        "Leiten Sie diesen Fall an das richtige Team weiter.",
    ),
}

OOD_DEPARTMENTS: tuple[str, ...] = ("Facilities", "Procurement")
OOD_TEMPLATES: dict[str, dict[str, list[str]]] = {
    "en": {
        "Facilities": [
            "The office badge reader is broken and nobody can enter the building.",
            "The air conditioning on the third floor has been off for a week.",
        ],
        "Procurement": [
            "We need to onboard a new supplier for office supplies.",
            "The purchase order for laptops is stuck in approval.",
        ],
    },
    "it": {
        "Facilities": ["Il lettore dei badge è rotto e nessuno può entrare in ufficio."],
        "Procurement": ["Dobbiamo inserire un nuovo fornitore per il materiale da ufficio."],
    },
}
GARBLED_STATES = (
    "asdkjh askjdh kjashd kajshd qweury",
    "the thing does not work with the other thing please fix asap",
)


@dataclass(frozen=True)
class SyntheticConfig:
    n_train: int = 1200
    n_val: int = 300
    n_test: int = 300
    n_ood: int = 100
    seed: int = 7
    min_candidates: int = 4
    max_candidates: int = 6
    difficulty_weights: tuple[float, float, float] = (0.40, 0.35, 0.25)
    multilingual_share_train: float = 0.15
    non_english_test_share: float = 0.5
    non_english_val_share: float = 0.35


@dataclass
class SyntheticSplits:
    train: DecisionDataset
    val: DecisionDataset
    test: DecisionDataset
    ood: DecisionDataset
    config: SyntheticConfig


def _pick_language(rng: random.Random, languages: Sequence[str], weights: Sequence[float]) -> str:
    return rng.choices(list(languages), weights=list(weights), k=1)[0]


def _build_candidate_set(
    rng: random.Random, department: str, difficulty: str, config: SyntheticConfig
) -> list[str]:
    target = rng.randint(config.min_candidates, config.max_candidates)
    candidates = [department]
    hard_pool = [d for d in HARD_NEGATIVES[department] if d != department]
    n_hard = 1 if difficulty == "hard" else (1 if rng.random() < 0.5 else 0)
    rng.shuffle(hard_pool)
    candidates += hard_pool[:n_hard]
    others = [d for d in DEPARTMENTS if d not in candidates]
    rng.shuffle(others)
    candidates += others[: max(0, target - len(candidates))]
    rng.shuffle(candidates)
    return candidates


def _make_example(rng: random.Random, language: str, difficulty: str, config: SyntheticConfig) -> DecisionExample:
    department = rng.choice(DEPARTMENTS)
    state = rng.choice(TEMPLATES[language][department][difficulty])
    question = rng.choice(QUESTIONS[language])
    candidates = _build_candidate_set(rng, department, difficulty, config)
    return DecisionExample(
        state=state,
        question=question,
        candidates=candidates,
        correct=department,
        metadata={"language": language, "difficulty": difficulty, "department": department},
    )


def _make_ood_example(rng: random.Random, config: SyntheticConfig) -> DecisionExample:
    language = rng.choice(("en", "en", "it"))
    department = rng.choice(OOD_DEPARTMENTS)
    if rng.random() < 0.15:
        state = rng.choice(GARBLED_STATES)
    else:
        state = rng.choice(OOD_TEMPLATES[language][department])
    others = list(DEPARTMENTS)
    rng.shuffle(others)
    target = rng.randint(config.min_candidates, config.max_candidates)
    candidates = [department] + others[: target - 1]
    rng.shuffle(candidates)
    return DecisionExample(
        state=state,
        question=rng.choice(QUESTIONS[language]),
        candidates=candidates,
        correct=department,
        metadata={"language": language, "difficulty": "ood", "department": department},
    )


def generate_examples(
    n: int, *, split: str, config: SyntheticConfig, ood: bool = False
) -> list[DecisionExample]:
    rng = random.Random(f"{config.seed}:{split}")
    if ood:
        return [_make_ood_example(rng, config) for _ in range(n)]
    non_english = {
        "train": config.multilingual_share_train,
        "val": config.non_english_val_share,
        "test": config.non_english_test_share,
    }.get(split, 0.3)
    examples: list[DecisionExample] = []
    for _ in range(n):
        language = "en" if rng.random() >= non_english else rng.choice(("it", "es", "fr", "de"))
        difficulty = rng.choices(DIFFICULTIES, weights=config.difficulty_weights, k=1)[0]
        examples.append(_make_example(rng, language, difficulty, config))
    return examples


def generate_dataset(config: SyntheticConfig | None = None, **overrides) -> SyntheticSplits:
    base = config or SyntheticConfig()
    if overrides:
        fields = {f.name for f in base.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        base = type(base)(**{**{f: getattr(base, f) for f in fields}, **overrides})
    return SyntheticSplits(
        train=DecisionDataset(generate_examples(base.n_train, split="train", config=base)),
        val=DecisionDataset(generate_examples(base.n_val, split="val", config=base)),
        test=DecisionDataset(generate_examples(base.n_test, split="test", config=base)),
        ood=DecisionDataset(generate_examples(base.n_ood, split="ood", config=base, ood=True)),
        config=base,
    )
