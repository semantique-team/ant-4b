"""Hand-written seed cases for the multi-use-case ANT dataset.

Each use case is one kind of decision (route a ticket, rate an incident,
approve a refund, ...). A label owns a handful of distinct situations (the
"states") plus optional aliases that the generator may use as the candidate
text instead of the canonical label, so the model learns to score candidate
*text* rather than memorise a closed label set.

The last few states of every label are deliberately hard: they mention a
confusable label without being owned by it.

Use cases flagged ``held_out`` never appear in train/val/test; they form the
OOD split that measures transfer to decisions the model has never seen.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Label:
    states: tuple[str, ...]
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class UseCase:
    name: str
    title: str
    questions: tuple[str, ...]
    labels: dict[str, Label]
    language: str = "en"
    held_out: bool = False


USE_CASES: tuple[UseCase, ...] = (
    UseCase(
        name="support_routing",
        title="Support routing",
        questions=(
            "Which team should handle this customer request?",
            "Who should this ticket be assigned to?",
            "Route this message to the right department.",
        ),
        labels={
            "Billing": Label(
                (
                    "I was charged twice for my monthly subscription and need one of the charges refunded.",
                    "The invoice total does not match the price in my order confirmation.",
                    "My card keeps being billed even though I cancelled the plan last month.",
                    "Can you send me a VAT invoice for the payment I made on the 3rd?",
                    "The sales rep promised a 20% discount but the invoice shows the full price.",
                    "After the upgrade I was billed for both the old and the new plan.",
                ),
                ("Billing team", "Finance"),
            ),
            "Technical Support": Label(
                (
                    "The mobile app crashes every time I open the reports page.",
                    "I cannot log in: the session expires right after I enter my password.",
                    "Exports from the dashboard produce a corrupted file that will not open.",
                    "The API returns a 500 error whenever I send a request with an attachment.",
                    "Customer success set up my workspace, but now the integrations page will not load.",
                    "Since the billing plan changed, single sign-on rejects every user on my team.",
                ),
                ("Tech support", "Engineering support"),
            ),
            "Sales": Label(
                (
                    "I would like a demo of the enterprise plan for a team of forty people.",
                    "What discount can we get if we buy two hundred seats?",
                    "We are comparing vendors and need a commercial quote before the end of the month.",
                    "What is the price difference between the Pro and Business tiers?",
                    "Billing says our plan covers ten users, but we want fifty and need new pricing.",
                    "Support fixed our usage limits and now we would like to discuss a larger contract.",
                ),
                ("Sales team", "Account executive"),
            ),
            "Legal": Label(
                (
                    "We received a subpoena asking for user data in a court case.",
                    "Our counsel needs the current data processing agreement before we sign the renewal.",
                    "A lawyer sent a formal notice claiming our product infringes a patent.",
                    "A competitor says our ads misuse their trademark and threatens action.",
                    "The security team confirmed a breach and we must notify the authority within 72 hours.",
                    "A customer cites GDPR rights and refuses the updated privacy terms.",
                ),
                ("Legal department", "Counsel"),
            ),
            "Customer Success": Label(
                (
                    "Our onboarding has stalled and nobody has contacted us in two weeks.",
                    "I want to schedule the quarterly business review that came with our plan.",
                    "How can we get training for the ten people we just hired?",
                    "We use only a fraction of the features we pay for and need an adoption plan.",
                    "The app kept crashing during our onboarding workshop, so the training could not finish.",
                    "We were double billed last month, and while that is fixed we also want to revisit our renewal goals.",
                ),
                ("Account management", "CS team"),
            ),
            "Security": Label(
                (
                    "I received a phishing email that impersonates your company.",
                    "Our SOC 2 audit needs evidence that multi-factor authentication is enforced.",
                    "Someone is trying to brute-force our admin account.",
                    "We found a vulnerability in your login flow and want to report it.",
                    "Legal asked for the chain of custody after the credential stuffing attempt we reported.",
                    "The phishing campaign targeted our billing data and we need forensic confirmation of what leaked.",
                ),
                ("Security team", "InfoSec"),
            ),
        },
    ),
    UseCase(
        name="incident_severity",
        title="Incident severity",
        questions=(
            "What is the severity of this incident?",
            "Which priority level should this incident get?",
            "How urgent is this issue?",
        ),
        labels={
            "P1": Label(
                (
                    "The checkout is down for every customer and no payment can be completed.",
                    "The production database lost its primary and writes are failing across all regions.",
                    "Customer personal data is publicly readable through an unauthenticated endpoint.",
                    "Login fails for 100% of users after the last deploy.",
                    "The main API has returned 5xx for 20 minutes and our largest customers are escalating.",
                ),
                ("Critical", "SEV1"),
            ),
            "P2": Label(
                (
                    "About 7% of EU payments time out and the rest succeed.",
                    "Search is very slow for customers in one region, but results still load.",
                    "The nightly export failed and the finance team needs the file by noon.",
                    "One availability zone is degraded; traffic is being served from the others.",
                    "Push notifications are delayed by about 30 minutes for all users.",
                ),
                ("High", "SEV2"),
            ),
            "P3": Label(
                (
                    "The settings page shows the wrong timezone label for some users.",
                    "A report column is misaligned in the PDF export.",
                    "An internal admin tool is slow, but the team can work around it.",
                    "One customer reports a duplicated email notification now and then.",
                    "A low-traffic endpoint returns a 404 instead of a 400 on bad input.",
                ),
                ("Medium", "SEV3"),
            ),
            "P4": Label(
                (
                    "A typo in the footer of the marketing site.",
                    "Request to rename a button in the admin dashboard for consistency.",
                    "The dark theme has a slightly wrong grey on one tooltip.",
                    "Suggestion to add a keyboard shortcut to the search box.",
                    "A deprecated log line is noisy but harmless.",
                ),
                ("Low", "SEV4"),
            ),
        },
    ),
    UseCase(
        name="sentiment",
        title="Message sentiment",
        questions=(
            "What is the sentiment of this message?",
            "How does the author feel?",
            "Classify the tone of this review.",
        ),
        labels={
            "positive": Label(
                (
                    "The new release is fantastic, everything feels faster and cleaner.",
                    "Support solved my problem in five minutes. Great service!",
                    "I was skeptical at first, but this tool saved my team hours every week.",
                    "Delivery was quick and the packaging was lovely.",
                    "Not perfect, but easily the best option I have tried so far.",
                ),
                ("Positive", "happy"),
            ),
            "negative": Label(
                (
                    "This update broke my workflow and nobody answers my emails.",
                    "Terrible experience: late, damaged, and the refund took a month.",
                    "I am really disappointed, it stopped working after two days.",
                    "The interface is confusing and the support bot just loops.",
                    "At first I loved it, but the constant crashes made me give up.",
                ),
                ("Negative", "unhappy"),
            ),
            "neutral": Label(
                (
                    "The package arrived on Tuesday and contains the items listed on the invoice.",
                    "I would like to know whether the plan includes API access.",
                    "The meeting is moved to 3 pm in room B.",
                    "The product has a 12 month warranty and ships from the EU warehouse.",
                    "Please find the updated specification attached.",
                ),
                ("Neutral", "mixed or neutral"),
            ),
        },
    ),
    UseCase(
        name="refund_policy",
        title="Refund approval",
        questions=(
            "Should this refund be approved?",
            "Does this refund request satisfy the policy?",
            "Decide on the refund request.",
        ),
        labels={
            "approve": Label(
                (
                    "Order delivered 5 days ago, item arrived broken, customer attached photos. Policy: refunds within 30 days, damaged items always covered.",
                    "Customer asks to return an unopened product 12 days after delivery. Policy: 30-day returns on unopened items.",
                    "Subscription was renewed yesterday and the customer never logged in. Policy: full refund within 7 days of renewal if unused.",
                    "Item never arrived; tracking shows it was returned to the sender. Policy: lost shipments are refunded in full.",
                    "The customer was billed twice for the same order. Policy: duplicate charges are always refunded.",
                ),
                ("yes", "Approve the refund"),
            ),
            "deny": Label(
                (
                    "The customer wants €340 back for an order shipped 45 days ago. Policy: refunds within 30 days only.",
                    "The product was used for five months and the customer says they simply changed their mind. Policy: no refunds on used items without a defect.",
                    "The customer asks for a refund on a digital download that was already used. Policy: digital goods are non-refundable after download.",
                    "A gift card purchased last year; customer wants cash back. Policy: gift cards are non-refundable.",
                    "Request after the 30-day window, although the customer is a loyal account holder. Policy has no exception for tenure.",
                ),
                ("no", "Deny the refund"),
            ),
            "escalate to manager": Label(
                (
                    "Refund of €4,800 requested for an annual contract after 40 days. Policy: manager approval required above €1,000.",
                    "The customer claims a defect, but there is no evidence and the item is 35 days old. Policy: ambiguous cases go to a manager.",
                    "Customer threatens legal action over a refund that falls outside the window. Policy: legal threats are handled by a manager.",
                    "A VIP account asks for an exception on a 31-day-old order. Policy: exceptions need manager sign-off.",
                    "The refund amount is within policy but the account was flagged for repeated refund abuse.",
                ),
                ("Send to a manager", "Needs manager review"),
            ),
        },
    ),
    UseCase(
        name="email_action",
        title="Email triage",
        questions=(
            "What should I do with this email?",
            "Choose the next action for this message.",
            "How should this email be handled?",
        ),
        labels={
            "reply now": Label(
                (
                    "My manager writes: 'Can you confirm today that the demo will be ready for Friday?'",
                    "A customer asks a quick yes/no question about the delivery date and is waiting for an answer.",
                    "The recruiter needs your availability for tomorrow before 5 pm to schedule the final interview.",
                    "A teammate is blocked and asks which of two API designs to use.",
                ),
                ("Reply immediately", "respond today"),
            ),
            "schedule a meeting": Label(
                (
                    "A partner proposes to discuss a possible integration and asks for a call next week.",
                    "The email thread has gone back and forth ten times on the roadmap and nobody agrees.",
                    "A new client wants an intro call to understand how we would work together.",
                    "The team asks for a retrospective after the failed release.",
                ),
                ("Set up a call", "book time with the sender"),
            ),
            "archive": Label(
                (
                    "Weekly newsletter from a SaaS vendor you used once.",
                    "Automated confirmation that your report finished exporting.",
                    "Receipt for the coffee machine ordered last month, already paid and filed.",
                    "A thank-you note that needs no answer.",
                ),
                ("Archive it", "no action needed"),
            ),
            "forward to a colleague": Label(
                (
                    "A customer writes about a billing mistake; billing is handled by Marta's team.",
                    "A candidate sends a CV for a data engineering role that you do not hire for.",
                    "A vendor asks who is in charge of the security questionnaire, which belongs to the CISO.",
                    "A press inquiry arrives in your inbox, but communications handles all media.",
                ),
                ("Forward it", "hand off to someone else"),
            ),
        },
    ),
    UseCase(
        name="order_intent",
        title="Shopper intent",
        questions=(
            "What does the customer want to do?",
            "Which action matches the customer's intent?",
            "Identify the intent of this chat message.",
        ),
        labels={
            "track order": Label(
                (
                    "Where is my package? It was supposed to arrive yesterday.",
                    "Can you tell me the status of order 48213?",
                    "The tracking page has not updated in four days.",
                    "When will my parcel reach me?",
                ),
                ("Check order status", "order tracking"),
            ),
            "cancel order": Label(
                (
                    "I ordered the wrong size, please cancel it before it ships.",
                    "Stop my order, I found it cheaper elsewhere.",
                    "I would like to cancel everything from this morning.",
                    "I placed the order twice by mistake, cancel the second one.",
                ),
                ("Cancel the order", "order cancellation"),
            ),
            "change address": Label(
                (
                    "I moved last week, can the delivery go to my new address instead?",
                    "The street number on my order is wrong, it should be 14 not 41.",
                    "Please deliver it to my office, not to my home.",
                    "I will be away, can you ship it to my mother's house?",
                ),
                ("Update shipping address", "edit delivery address"),
            ),
            "return item": Label(
                (
                    "The shoes do not fit, how do I send them back?",
                    "I want to return the blender, it is not what I expected.",
                    "Do I get a label to return a defective headset?",
                    "This is the wrong colour; I would like to exchange it.",
                ),
                ("Start a return", "exchange or return"),
            ),
            "talk to a human": Label(
                (
                    "This bot is useless, put me through to a real person.",
                    "I need to speak with an agent about a complicated issue.",
                    "Can someone call me? This is too long to type.",
                    "Human please.",
                ),
                ("Transfer to an agent", "human agent"),
            ),
        },
    ),
    UseCase(
        name="content_moderation",
        title="Content moderation",
        questions=(
            "What should moderation do with this post?",
            "Does this content violate the community guidelines?",
            "Choose the moderation action.",
        ),
        labels={
            "allow": Label(
                (
                    "Does anyone know a good tutorial for learning Rust?",
                    "I strongly disagree with the new pricing, and here is why.",
                    "Photo of my dog at the beach this weekend.",
                    "The referee was awful yesterday; that was never a penalty.",
                ),
                ("Approve", "leave it up"),
            ),
            "flag for review": Label(
                (
                    "If this keeps up somebody is going to regret it.",
                    "Join my crypto group and double your money in a week, link in bio.",
                    "A post with a graphic accident photo without a content warning.",
                    "Comment that quotes a slur while arguing it should be banned.",
                ),
                ("Send to a human moderator", "needs review"),
            ),
            "remove": Label(
                (
                    "I will find out where you live and make you pay.",
                    "Selling stolen credit card numbers, message me.",
                    "Sharing a private person's home address and phone number to harass them.",
                    "Explicit hate speech targeting a protected group.",
                ),
                ("Remove the post", "take it down"),
            ),
        },
    ),
    UseCase(
        name="expense_category",
        title="Expense category",
        questions=(
            "Which category does this expense belong to?",
            "Where should this expense be booked?",
            "Categorise this receipt.",
        ),
        labels={
            "Travel": Label(
                (
                    "Train ticket Milan to Rome for the client workshop.",
                    "Two nights at a hotel near the conference venue.",
                    "Taxi from the airport to the office after a late flight.",
                    "Car rental for the on-site installation in Turin.",
                ),
                ("Travel and transport",),
            ),
            "Meals": Label(
                (
                    "Dinner with a client after the contract signing.",
                    "Team lunch to celebrate the release.",
                    "Coffee and pastries for the Monday workshop attendees.",
                    "Sandwiches ordered for the all-hands.",
                ),
                ("Meals and entertainment",),
            ),
            "Software": Label(
                (
                    "Annual licence for the design tool, 12 seats.",
                    "Monthly invoice for the cloud monitoring service.",
                    "One-year subscription to the code hosting platform.",
                    "Domain renewal and TLS certificate.",
                ),
                ("Software and subscriptions", "SaaS"),
            ),
            "Office supplies": Label(
                (
                    "Ink cartridges and printer paper for the second floor.",
                    "Ergonomic chair for a new hire.",
                    "Whiteboard markers and sticky notes.",
                    "Replacement keyboard and mouse for a broken desk setup.",
                ),
                ("Office equipment",),
            ),
            "Training": Label(
                (
                    "Registration for the cloud security certification exam.",
                    "Online course on machine learning for the data team.",
                    "Ticket for a two-day developer conference, learning sessions only.",
                    "Subscription to a language learning platform for the sales team.",
                ),
                ("Learning and development",),
            ),
        },
    ),
    UseCase(
        name="bug_component",
        title="Bug triage",
        questions=(
            "Which component owns this bug?",
            "Where does this bug belong?",
            "Assign this bug to a team.",
        ),
        labels={
            "frontend": Label(
                (
                    "The dropdown does not close after selecting an option in Safari.",
                    "Button label is cut off on narrow screens.",
                    "The form loses its values when I switch tabs.",
                    "Dark mode shows white text on a white card in the settings modal.",
                ),
                ("web UI", "client side"),
            ),
            "backend": Label(
                (
                    "The /orders endpoint returns duplicate rows when two filters are combined.",
                    "Password reset emails are queued but never sent.",
                    "Rate limiting counts failed requests twice.",
                    "The webhook retry job crashes on payloads over 1 MB.",
                ),
                ("API service", "server side"),
            ),
            "database": Label(
                (
                    "The migration to add the invoices index times out on production.",
                    "A query on the events table takes 40 seconds since it passed 100M rows.",
                    "Deadlocks appear whenever two imports run together.",
                    "Replica lag exceeds ten minutes during the nightly batch.",
                ),
                ("data layer", "DBA team"),
            ),
            "infrastructure": Label(
                (
                    "Pods keep restarting with OOMKilled after the last release.",
                    "The TLS certificate on the staging load balancer expired.",
                    "CI runners are stuck in the queue for over an hour.",
                    "Autoscaling does not add nodes when the CPU is above 90%.",
                ),
                ("platform / DevOps", "cloud infra"),
            ),
            "mobile": Label(
                (
                    "The iOS app freezes on launch after updating to the latest OS.",
                    "Push notification taps open the home screen, not the message.",
                    "The Android back button closes the app from the checkout.",
                    "Biometric login fails on tablets only.",
                ),
                ("mobile apps",),
            ),
        },
    ),
    UseCase(
        name="lead_quality",
        title="Lead qualification",
        questions=(
            "How qualified is this lead?",
            "What temperature is this lead?",
            "Rate this sales lead.",
        ),
        labels={
            "hot": Label(
                (
                    "The CTO of a 500-person company filled in the demo form and asked for pricing for 300 seats this quarter.",
                    "Procurement contacted us with an approved budget and wants a contract draft by Friday.",
                    "A returning customer asks to double their licences before the end of the month.",
                ),
                ("Hot lead", "ready to buy"),
            ),
            "warm": Label(
                (
                    "A product manager downloaded the whitepaper and attended the webinar.",
                    "The lead says the company will review vendors in six months and asks for a feature list.",
                    "An engineer from a mid-size firm asks how the API compares with a competitor.",
                ),
                ("Warm lead", "interested"),
            ),
            "cold": Label(
                (
                    "A student subscribed to the newsletter with a personal address.",
                    "A company with three employees, outside our target markets, clicked an ad once.",
                    "Someone requested a brochure, and gave no company or role.",
                ),
                ("Cold lead", "not qualified"),
            ),
        },
    ),
    UseCase(
        name="document_type",
        title="Document type",
        questions=(
            "What type of document is this?",
            "Classify this document.",
            "Which folder does this file belong in?",
        ),
        labels={
            "invoice": Label(
                (
                    "Invoice no. 2024-118, due in 30 days, bank transfer details at the bottom, total including VAT.",
                    "Please pay the attached amount for services rendered in March, reference PO 7781.",
                    "Amount due: 1,250.00 EUR. Payment terms: net 30. IBAN below.",
                ),
                ("Invoice",),
            ),
            "contract": Label(
                (
                    "This agreement is entered into between the parties; the term is 24 months, renewable; governing law is Italian law.",
                    "Section 4: confidentiality. Each party shall keep the other's information secret for five years.",
                    "Signed by the authorised representatives on the date written above.",
                ),
                ("Contract", "legal agreement"),
            ),
            "resume": Label(
                (
                    "Experience: 6 years as backend engineer, Python and Go, led a team of four. Education: MSc in Computer Science.",
                    "Skills: React, TypeScript, testing. Languages: English, Italian. Available from September.",
                    "Objective: to join a data team where I can grow as a machine learning engineer.",
                ),
                ("CV", "Resume"),
            ),
            "receipt": Label(
                (
                    "Thank you for your purchase. Card ending 4421. Total paid 23.80. Receipt 009124.",
                    "Cash register slip: 2x coffee, 1x croissant, total 5.40, paid.",
                    "Payment received, order #5521, no further action needed.",
                ),
                ("Receipt",),
            ),
            "report": Label(
                (
                    "Q3 results: revenue grew 12% year over year, churn fell to 2.1%, headcount is flat.",
                    "Executive summary: the pilot reduced handling time by 30%, with recommendations on page 4.",
                    "Findings of the audit and the corrective actions planned for next quarter.",
                ),
                ("Report",),
            ),
        },
    ),
    UseCase(
        name="transaction_risk",
        title="Transaction risk",
        questions=(
            "What should we do with this transaction?",
            "Choose the fraud decision for this payment.",
            "Is this payment safe to process?",
        ),
        labels={
            "approve": Label(
                (
                    "EUR 42 at a supermarket in the cardholder's home city, chip and PIN, usual weekly pattern.",
                    "Monthly 9.99 subscription renewal to a merchant used for two years.",
                    "EUR 120 at a pharmacy 2 km from home, 10 minutes after a normal transaction.",
                ),
                ("Approve", "allow"),
            ),
            "manual review": Label(
                (
                    "EUR 1,900 on electronics, first time at this merchant, delivery to a new address.",
                    "A purchase in another country two hours after one at home, but the customer previously informed us of travel.",
                    "A card used for 6 small online purchases in 15 minutes at new merchants.",
                ),
                ("Review manually", "hold for analyst"),
            ),
            "block": Label(
                (
                    "EUR 4,200 gift cards bought from a foreign IP, card reported stolen yesterday.",
                    "Three declined attempts followed by a small test charge, then a large charge on a different continent.",
                    "Transaction from a device flagged in a known fraud ring, billing and shipping mismatch.",
                ),
                ("Block", "decline"),
            ),
        },
    ),
    UseCase(
        name="it_helpdesk",
        title="IT helpdesk",
        questions=(
            "What kind of IT request is this?",
            "Which queue should this IT ticket go to?",
            "Classify this helpdesk ticket.",
        ),
        labels={
            "password reset": Label(
                (
                    "I forgot my laptop password and I am locked out.",
                    "My account is locked after too many attempts.",
                    "The two-factor code is not arriving, I cannot sign in to the portal.",
                ),
                ("Credentials",),
            ),
            "hardware": Label(
                (
                    "The screen of my laptop flickers and sometimes goes black.",
                    "My docking station does not detect the second monitor.",
                    "The keyboard has three dead keys after a coffee spill.",
                ),
                ("Hardware issue",),
            ),
            "software install": Label(
                (
                    "I need Docker installed for the new project.",
                    "Can you add the PDF editing tool to my machine?",
                    "The IDE needs a newer version and I do not have admin rights.",
                ),
                ("Install software",),
            ),
            "network": Label(
                (
                    "Wi-Fi drops every few minutes in the east wing.",
                    "I cannot connect to the VPN from home.",
                    "Shared drive is unreachable but the internet works fine.",
                ),
                ("Network problem",),
            ),
            "access request": Label(
                (
                    "Please give me read access to the analytics dashboard, my manager approved it.",
                    "New hire needs to be added to the engineering GitHub organisation.",
                    "I need permission to the finance folder for the audit.",
                ),
                ("Permissions",),
            ),
        },
    ),
    UseCase(
        name="agent_next_step",
        title="Agent next step",
        questions=(
            "What should the assistant do next?",
            "Pick the next step for the agent.",
            "Which action fits the current situation?",
        ),
        labels={
            "answer directly": Label(
                (
                    "The user asks what the capital of France is. No tools are needed.",
                    "The user asks for a synonym of 'quick'.",
                    "The user asks you to explain the difference between a list and a tuple in Python.",
                ),
                ("Reply without tools",),
            ),
            "call a tool": Label(
                (
                    "The user asks for today's weather in Rome and a weather tool is available.",
                    "The user wants the current stock price of a company; a market data tool is available.",
                    "The user asks to book a meeting room and the calendar tool can do it.",
                ),
                ("Use a tool", "call the API"),
            ),
            "ask for clarification": Label(
                (
                    "The user says 'send it to him' without saying who or what.",
                    "The user asks to 'fix the report' but three reports are open.",
                    "The user wants a flight to 'Paris', and there is also a Paris in Texas.",
                ),
                ("Ask a follow-up question",),
            ),
            "refuse": Label(
                (
                    "The user asks for step-by-step instructions to break into someone else's account.",
                    "The user wants a fake doctor's note to hand to an employer.",
                    "The user asks the assistant to write malware that encrypts a victim's files.",
                ),
                ("Decline the request",),
            ),
        },
    ),
    UseCase(
        name="support_routing_it",
        title="Instradamento (italiano)",
        language="it",
        questions=(
            "Quale team deve gestire questa richiesta?",
            "A chi va assegnato questo ticket?",
        ),
        labels={
            "Billing": Label(
                (
                    "Sono stato addebitato due volte per lo stesso abbonamento e chiedo un rimborso.",
                    "L'importo della fattura non corrisponde all'ordine.",
                    "Mi addebitate ancora la carta anche se ho disdetto il piano.",
                ),
            ),
            "Technical Support": Label(
                (
                    "L'app si blocca ogni volta che apro la pagina dei report.",
                    "Non riesco ad accedere: la sessione scade subito dopo la password.",
                    "L'esportazione dalla dashboard produce un file corrotto.",
                ),
            ),
            "Sales": Label(
                (
                    "Vorrei una demo del piano enterprise per quaranta persone.",
                    "Che sconto possiamo avere per duecento postazioni?",
                    "Ci serve un preventivo prima della fine del mese.",
                ),
            ),
            "Legal": Label(
                (
                    "Abbiamo ricevuto una diffida formale per violazione di brevetto.",
                    "Il nostro avvocato chiede l'accordo sul trattamento dei dati prima del rinnovo.",
                    "Un concorrente minaccia azioni legali per l'uso del suo marchio.",
                ),
            ),
        },
    ),
    UseCase(
        name="sentiment_it",
        title="Sentiment (italiano)",
        language="it",
        questions=("Qual è il sentimento di questo messaggio?", "Com'è il tono dell'autore?"),
        labels={
            "positive": Label(
                (
                    "Servizio eccellente, problema risolto in cinque minuti!",
                    "Prodotto bellissimo, lo consiglio a tutti.",
                    "Consegna rapidissima e imballaggio curato.",
                ),
            ),
            "negative": Label(
                (
                    "Esperienza pessima: in ritardo, danneggiato e nessuno risponde.",
                    "Sono molto deluso, ha smesso di funzionare dopo due giorni.",
                    "Il supporto non mi ha aiutato e ho perso un'ora.",
                ),
            ),
            "neutral": Label(
                (
                    "Il pacco è arrivato martedì con gli articoli della fattura.",
                    "Vorrei sapere se il piano include l'accesso alle API.",
                    "La riunione è spostata alle 15 nella sala B.",
                ),
            ),
        },
    ),
    UseCase(
        name="meeting_action",
        title="Meeting action",
        questions=(
            "What should happen after this meeting?",
            "Which follow-up action fits this meeting note?",
            "Pick the next step for the meeting owner.",
        ),
        labels={
            "send notes": Label(
                (
                    "The meeting covered decisions and owners but no recap was sent.",
                    "The workshop ended with agreed action items and attendees asked for a summary.",
                    "The demo concluded and the customer requested the recording and next steps.",
                    "The weekly sync confirmed dates and owners but nobody documented them.",
                    "The review call ended with three decisions that the team needs in writing.",
                ),
                ("Share the recap", "Send minutes"),
            ),
            "schedule follow-up": Label(
                (
                    "The customer asked to continue the discussion next week but no date was chosen.",
                    "The technical review found open questions that need another working session.",
                    "The negotiation call ended with a request for a follow-up demo.",
                    "The manager asked to revisit staffing after the pilot results.",
                    "The design review left the pricing decision to a later meeting.",
                ),
                ("Book the next call",),
            ),
            "create task": Label(
                (
                    "The meeting assigned three deliverables with owners and due dates.",
                    "The retro identified a process fix that should be tracked.",
                    "The planning call produced a list of engineering work for the next sprint.",
                    "The customer success review agreed to prepare an adoption report.",
                    "The audit walkthrough found evidence that someone must collect.",
                ),
                ("Open a ticket", "Track the action"),
            ),
            "escalate": Label(
                (
                    "The meeting revealed a legal deadline that the team cannot meet.",
                    "The customer threatened to churn and asked for executive sponsorship.",
                    "The project review found a budget overrun requiring a director decision.",
                    "The security discussion uncovered an incident that must be reported.",
                    "The steering committee disagreed on scope and asked for an executive ruling.",
                ),
                ("Escalate to management",),
            ),
        },
    ),
    UseCase(
        name="invoice_discrepancy",
        title="Invoice discrepancy",
        questions=(
            "How should this invoice discrepancy be handled?",
            "What is the next step for this invoice?",
            "Classify this accounts-payable issue.",
        ),
        labels={
            "match": Label(
                (
                    "The purchase order, goods receipt and invoice show the same amount and vendor.",
                    "The subscription invoice matches the contracted seat count and price.",
                    "The bank payment matches the invoice total after the agreed discount.",
                    "The expense report receipt and submitted amount are identical.",
                    "The utility invoice follows the fixed monthly rate in the agreement.",
                ),
                ("Approve payment", "No discrepancy"),
            ),
            "request clarification": Label(
                (
                    "The invoice lists a service period that the contract does not define.",
                    "The receipt is missing the VAT number and the amount cannot be checked.",
                    "The invoice has two line items but the purchase order has one.",
                    "The vendor name differs from the contract and no reference is provided.",
                    "The invoice quantity is blank and the total cannot be reproduced.",
                ),
                ("Ask the vendor",),
            ),
            "flag finance": Label(
                (
                    "The invoice is 18% higher than the approved purchase order.",
                    "The same invoice number appears twice in the ledger.",
                    "The payment term is 60 days while the contract says 30.",
                    "The bank details changed just before the payment run.",
                    "The invoice includes an unapproved travel expense.",
                ),
                ("Hold for review",),
            ),
            "escalate": Label(
                (
                    "The invoice claims a penalty that legal has not accepted.",
                    "The vendor threatens collection action over a disputed amount.",
                    "The invoice is tied to a suspected fraud report.",
                    "A director approved an invoice that bypassed procurement.",
                    "The discrepancy involves a regulated payment and cannot be corrected locally.",
                ),
                ("Escalate to finance lead",),
            ),
        },
    ),
    UseCase(
        name="data_privacy_request",
        title="Data privacy request",
        questions=(
            "How should this privacy request be handled?",
            "Which response fits this data-rights request?",
            "Classify the requested privacy action.",
        ),
        labels={
            "access": Label(
                (
                    "The user asks for a copy of all personal data the company holds.",
                    "A customer requests the history of their account activity.",
                    "The user asks which third parties received their data.",
                    "The employee requests a copy of their HR data.",
                    "The user wants to know what profiling data is stored.",
                ),
                ("Export the data",),
            ),
            "delete": Label(
                (
                    "The user asks to delete their account and all personal data.",
                    "The customer requests erasure after closing the subscription.",
                    "The user asks to remove a forum post containing their email address.",
                    "The ex-employee requests deletion of recruitment data.",
                    "The user withdraws consent for marketing data.",
                ),
                ("Erase the data",),
            ),
            "correct": Label(
                (
                    "The user says their name is misspelled in the invoice.",
                    "The customer reports an incorrect delivery address in their profile.",
                    "The user says their date of birth is wrong.",
                    "The account shows an old phone number.",
                    "The user disputes the stored company affiliation.",
                ),
                ("Fix the record",),
            ),
            "reject": Label(
                (
                    "The request asks to erase records the company must keep for tax law.",
                    "The user asks to delete data needed for an active fraud investigation.",
                    "The request is from someone who cannot verify ownership of the account.",
                    "The user asks to remove content belonging to another person.",
                    "The request would delete data required for a court case.",
                ),
                ("Decline the request",),
            ),
        },
    ),
    UseCase(
        name="project_status",
        title="Project status",
        questions=(
            "What is the status of this project?",
            "How should this project update be classified?",
            "Choose the project health label.",
        ),
        labels={
            "on track": Label(
                (
                    "The milestone is due Friday and the remaining work is already merged.",
                    "The team completed the planned sprint with no blocked items.",
                    "The budget and timeline match the approved baseline.",
                    "The customer accepted the prototype and the next release is scheduled.",
                    "All critical risks have owners and mitigation dates.",
                ),
                ("Green", "Healthy"),
            ),
            "at risk": Label(
                (
                    "The integration is late but the team has a workaround for the launch.",
                    "Two key dependencies are waiting on another team.",
                    "The scope grew after the last review without a change request.",
                    "Testing started late and the buffer is now one day.",
                    "The supplier confirmed a delivery slip that may affect the demo.",
                ),
                ("Amber", "Needs attention"),
            ),
            "blocked": Label(
                (
                    "The vendor credential expired and nobody can deploy to production.",
                    "The legal review is blocking the data export feature.",
                    "The environment is unavailable and the team cannot run tests.",
                    "A required approval has been waiting for two weeks.",
                    "The critical bug has no owner and the release cannot proceed.",
                ),
                ("Red", "Stopped"),
            ),
            "done": Label(
                (
                    "The acceptance criteria are signed off and the release notes are published.",
                    "The migration completed and the old system is decommissioned.",
                    "The customer confirmed the deliverable in writing.",
                    "All post-launch monitoring is green for seven days.",
                    "The project was closed after the final retrospective.",
                ),
                ("Complete", "Closed"),
            ),
        },
    ),
    UseCase(
        name="candidate_screening",
        title="Candidate screening",
        questions=(
            "What should happen to this job application?",
            "Which screening decision fits this candidate?",
            "Decide the next recruiting step.",
        ),
        labels={
            "advance": Label(
                (
                    "The candidate meets all required years and the first interview was strong.",
                    "The portfolio matches the role and the availability date is confirmed.",
                    "The referral says the candidate has led similar projects.",
                    "The technical test is above the bar and the salary expectation fits.",
                    "The candidate has the required certification and clear motivation.",
                ),
                ("Move to interview",),
            ),
            "hold": Label(
                (
                    "The application is promising but the availability date is unclear.",
                    "The resume is strong but the salary expectation is above the band.",
                    "The candidate has the skills but the interview schedule is not set.",
                    "The profile is interesting but the required language level is not proven.",
                    "The application needs a portfolio review before a decision.",
                ),
                ("Keep warm",),
            ),
            "reject": Label(
                (
                    "The candidate has no experience with the required technology.",
                    "The application is for a different location with no relocation plan.",
                    "The required certification is missing and cannot be obtained before the start date.",
                    "The interview showed a clear mismatch with the role responsibilities.",
                    "The candidate withdrew from the process.",
                ),
                ("Decline",),
            ),
            "request more information": Label(
                (
                    "The resume does not show the dates of the last two roles.",
                    "The application lacks links to the portfolio.",
                    "The candidate did not answer the screening questions.",
                    "The employment history has an unexplained gap.",
                    "The references are missing.",
                ),
                ("Ask for documents",),
            ),
        },
    ),
    UseCase(
        name="azione_email_it",
        title="Azione email (italiano)",
        language="it",
        questions=(
            "Cosa si deve fare con questa email?",
            "Quale azione è più adatta per questo messaggio?",
            "Decidi la prossima azione per questa email.",
        ),
        labels={
            "reply now": Label(
                (
                    "Il cliente chiede una conferma urgente prima delle 18.",
                    "Il fornitore ha inviato un preventivo con scadenza oggi.",
                    "Il capo chiede una risposta immediata su un problema in produzione.",
                    "Il candidato aspetta un feedback entro domani.",
                    "Il messaggio contiene un errore di pagamento da correggere subito.",
                ),
                ("Rispondi subito",),
            ),
            "schedule meeting": Label(
                (
                    "Il cliente vuole una call per approfondire l'offerta.",
                    "Il team deve discutere un progetto con più stakeholder.",
                    "La trattativa richiede una riunione con il legale.",
                    "Il manager propone un confronto sul piano del prossimo trimestre.",
                    "Serve un workshop per chiarire i requisiti.",
                ),
                ("Fissa un appuntamento",),
            ),
            "archive": Label(
                (
                    "Il messaggio contiene solo un aggiornamento informativo.",
                    "La newsletter non richiede alcuna azione.",
                    "La comunicazione è già stata gestita da un collega.",
                    "L'avviso di sistema non è rilevante per il destinatario.",
                    "Il promemoria riguarda un evento già concluso.",
                ),
                ("Archivia",),
            ),
            "forward to a colleague": Label(
                (
                    "La richiesta riguarda la fatturazione e non il supporto tecnico.",
                    "Il messaggio è destinato al team legale.",
                    "Il cliente chiede informazioni commerciali.",
                    "Il problema richiede l'intervento del responsabile sicurezza.",
                    "La domanda è per il reparto risorse umane.",
                ),
                ("Inoltra",),
            ),
        },
    ),
    UseCase(
        name="richiesta_assistenza_it",
        title="Richiesta di assistenza (italiano)",
        language="it",
        questions=(
            "Che tipo di richiesta è questa?",
            "In quale coda deve finire questo ticket?",
            "Classifica questa richiesta di assistenza.",
        ),
        labels={
            "tecnico": Label(
                (
                    "L'applicazione si blocca durante l'accesso.",
                    "Il server non risponde da stamattina.",
                    "L'aggiornamento ha provocato un errore di configurazione.",
                    "Non riesco a stampare dal portale.",
                    "L'integrazione con il CRM restituisce un errore.",
                ),
                ("Supporto tecnico",),
            ),
            "amministrativo": Label(
                (
                    "Devo aggiornare la ragione sociale sull'account.",
                    "Vorrei cambiare la modalità di pagamento.",
                    "Serve una copia della fattura di marzo.",
                    "Devo aggiungere un utente al contratto.",
                    "Vorrei sapere quando scade il rinnovo.",
                ),
                ("Pratiche amministrative",),
            ),
            "commerciale": Label(
                (
                    "Vorrei un preventivo per ampliare le licenze.",
                    "Chiedo informazioni sul piano enterprise.",
                    "Vorrei una demo per il mio team.",
                    "Qual è lo sconto per un rinnovo triennale?",
                    "Vorrei confrontare due piani di abbonamento.",
                ),
                ("Ufficio vendite",),
            ),
            "reclamo": Label(
                (
                    "Il servizio non corrisponde a quanto promesso.",
                    "Ho aspettato tre giorni senza risposta.",
                    "L'addebito non è stato autorizzato.",
                    "Il prodotto è arrivato danneggiato.",
                    "Vorrei parlare con un responsabile.",
                ),
                ("Gestione reclami",),
            ),
        },
    ),
    UseCase(
        name="urgenza_ticket_it",
        title="Urgenza ticket (italiano)",
        language="it",
        questions=(
            "Quale priorità ha questo ticket?",
            "Quanto è urgente questo problema?",
            "Assegna il livello di urgenza.",
        ),
        labels={
            "P1": Label(
                (
                    "Il pagamento non funziona per nessun cliente.",
                    "Il database principale non accetta scritture.",
                    "I dati sensibili sono accessibili senza autenticazione.",
                    "L'area riservata è fuori uso per tutti gli utenti.",
                    "L'API critica restituisce errori da mezz'ora.",
                ),
                ("Critico",),
            ),
            "P2": Label(
                (
                    "Una parte dei pagamenti va in timeout.",
                    "La ricerca è lentissima in una regione.",
                    "L'export notturno è fallito.",
                    "Le notifiche arrivano con mezz'ora di ritardo.",
                    "Un ambiente di produzione è degradato.",
                ),
                ("Alto",),
            ),
            "P3": Label(
                (
                    "Un report mostra una colonna fuori posto.",
                    "Una pagina admin è lenta ma funziona.",
                    "Alcuni utenti vedono un fuso orario errato.",
                    "Un avviso viene inviato due volte.",
                    "Un endpoint secondario restituisce un codice errato.",
                ),
                ("Medio",),
            ),
            "P4": Label(
                (
                    "C'è un refuso nel piè di pagina.",
                    "Va rinominato un pulsante.",
                    "Il tema scuro ha una tonalità leggermente diversa.",
                    "Manca una scorciatoia da tastiera.",
                    "Un log deprecato è rumoroso ma innocuo.",
                ),
                ("Basso",),
            ),
        },
    ),
    # --- held out: never seen in training, used for the OOD split ---
    UseCase(
        name="contract_review",
        title="Contract review",
        held_out=True,
        questions=("What should the reviewer do with this contract?", "Decide on this contract."),
        labels={
            "approve": Label(
                (
                    "The contract matches the approved template and the supplier accepted the standard terms.",
                    "The renewal keeps the same price and service level.",
                    "The NDA uses the company's standard clauses.",
                    "The statement of work is within the approved budget and scope.",
                    "The amendment only changes the billing address.",
                ),
                ("Sign",),
            ),
            "flag clause": Label(
                (
                    "The contract has unlimited liability for data loss.",
                    "The renewal includes an automatic price increase.",
                    "The supplier wants to own the deliverables created by the company.",
                    "The termination notice period is shorter than policy allows.",
                    "The governing law is outside the approved jurisdiction.",
                ),
                ("Mark for revision",),
            ),
            "escalate": Label(
                (
                    "The contract requires a security commitment the company cannot meet.",
                    "The supplier refuses the audit clause for a regulated customer.",
                    "The contract includes a non-compete affecting other clients.",
                    "The customer requests a bespoke indemnity for a public sector deal.",
                    "The deal has a liability cap above the delegated authority.",
                ),
                ("Send to legal",),
            ),
        },
    ),
    UseCase(
        name="approvazione_spese_it",
        title="Approvazione spese (italiano)",
        language="it",
        held_out=True,
        questions=("Come va gestita questa nota spese?", "Decidi sulla richiesta di rimborso."),
        labels={
            "approva": Label(
                (
                    "La spesa è entro budget e ha ricevuta valida.",
                    "Il viaggio era approvato prima della partenza.",
                    "La formazione rientra nel piano annuale.",
                    "Il materiale è necessario per un progetto già approvato.",
                    "L'importo è inferiore alla soglia del responsabile.",
                ),
                ("Rimborsa",),
            ),
            "rifiuta": Label(
                (
                    "La spesa non ha alcun documento di supporto.",
                    "Il ristorante supera il limite di policy.",
                    "L'acquisto è stato fatto senza autorizzazione.",
                    "La spesa è duplicata.",
                    "Il bene non è pertinente al lavoro.",
                ),
                ("Non rimborsare",),
            ),
            "chiedi informazioni": Label(
                (
                    "La ricevuta è illeggibile.",
                    "Il centro di costo non è indicato.",
                    "Il motivo del viaggio non è specificato.",
                    "La fattura è intestata a un'altra società.",
                    "Manca la data della spesa.",
                ),
                ("Chiedi chiarimenti",),
            ),
        },
    ),
    # --- older held out: never seen in training, used for the OOD split ---
    UseCase(
        name="pr_review",
        title="Pull request review",
        held_out=True,
        questions=("What should the reviewer do with this pull request?", "Decide on this PR."),
        labels={
            "approve and merge": Label(
                (
                    "Fixes a typo in the docs, CI is green, one-line change.",
                    "Adds a missing unit test for an existing helper; no production code changed.",
                    "Small bug fix with a regression test and a clear description; all checks pass.",
                ),
            ),
            "request changes": Label(
                (
                    "Adds a feature but has no tests and CI is red because of a lint error.",
                    "Changes 40 files including an unrelated refactor, and the description is empty.",
                    "Introduces a SQL query built from user input without escaping.",
                ),
            ),
            "discuss design first": Label(
                (
                    "Proposes replacing the message queue with a new technology; no prior design document.",
                    "Changes a public API signature that other teams depend on.",
                    "Adds a new service boundary, which conflicts with the agreed architecture.",
                ),
            ),
        },
    ),
    UseCase(
        name="shipping_method",
        title="Shipping method",
        held_out=True,
        questions=("Which shipping option fits this order?", "Pick the delivery method."),
        labels={
            "standard": Label(
                (
                    "A birthday gift ordered three weeks before the date; the customer wants the cheapest option.",
                    "Replacement cables, nothing urgent, free shipping threshold reached.",
                    "Books for next semester, which starts in two months.",
                ),
            ),
            "express": Label(
                (
                    "A laptop charger needed for a trip in three days.",
                    "The customer needs the replacement part by the end of the week for a repair.",
                    "A gift for an anniversary happening in four days.",
                ),
            ),
            "overnight": Label(
                (
                    "The only copy of a signed contract must arrive tomorrow morning.",
                    "A lab sample kit needed for an experiment starting at 8 am tomorrow.",
                    "A wedding dress alteration needed for a ceremony the day after tomorrow, courier cutoff is tonight.",
                ),
            ),
        },
    ),
    UseCase(
        name="leave_request",
        title="Leave request",
        held_out=True,
        questions=("How should HR respond to this leave request?", "Decide on the time-off request."),
        labels={
            "approve": Label(
                (
                    "Employee requests 3 days of annual leave next month with 22 days of balance; team coverage is arranged.",
                    "Employee requests one afternoon off for a medical appointment, with a note.",
                    "Employee requests two weeks in August, requested four months in advance, no clashes in the team.",
                ),
            ),
            "deny": Label(
                (
                    "Employee requests 15 days of leave but has only 4 days of balance left.",
                    "Employee requests leave during the quarter-end freeze, when all of the finance team is required.",
                    "Three of the four team members have already booked the same week and the minimum coverage is two.",
                ),
            ),
            "ask for more information": Label(
                (
                    "Employee requests 'some days off' without dates.",
                    "Employee asks for special leave for a family matter but does not say which type applies.",
                    "Employee requests unpaid leave for six months without explanation or a return date.",
                ),
            ),
        },
    ),
)
