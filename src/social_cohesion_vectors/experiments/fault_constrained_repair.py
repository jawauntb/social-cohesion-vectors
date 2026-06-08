"""Constrained repair candidate composition for hard availability failures."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from social_cohesion_vectors.experiments.fault_generation import FaultPromptRecord

CONSTRAINED_REPAIR_COMPOSER_VERSION = "constrained_repair_v1"
LEXICAL_BALANCED_REPAIR_COMPOSER_VERSION = "lexical_balanced_repair_v1"
SOURCE_DIVERSE_REPAIR_COMPOSER_VERSION = "source_diverse_repair_v1"
LEXICAL_ADVERSARIAL_REPAIR_COMPOSER_VERSION = "lexical_adversarial_repair_v1"
CROSS_FAULT_LEXICAL_REPAIR_COMPOSER_VERSION = "cross_fault_lexical_repair_v1"
CONSTRAINED_REPAIR_PROVIDER = "constrained_repair"
CONSTRAINED_REPAIR_MODEL = CONSTRAINED_REPAIR_COMPOSER_VERSION
LEXICAL_BALANCED_REPAIR_MODEL = LEXICAL_BALANCED_REPAIR_COMPOSER_VERSION
SOURCE_DIVERSE_REPAIR_MODEL = SOURCE_DIVERSE_REPAIR_COMPOSER_VERSION
LEXICAL_ADVERSARIAL_REPAIR_MODEL = LEXICAL_ADVERSARIAL_REPAIR_COMPOSER_VERSION
CROSS_FAULT_LEXICAL_REPAIR_MODEL = CROSS_FAULT_LEXICAL_REPAIR_COMPOSER_VERSION
CONSTRAINED_REPAIR_COMPOSER_VERSION_CHOICES: tuple[str, ...] = (
    CONSTRAINED_REPAIR_COMPOSER_VERSION,
    LEXICAL_BALANCED_REPAIR_COMPOSER_VERSION,
    SOURCE_DIVERSE_REPAIR_COMPOSER_VERSION,
    LEXICAL_ADVERSARIAL_REPAIR_COMPOSER_VERSION,
    CROSS_FAULT_LEXICAL_REPAIR_COMPOSER_VERSION,
)

CONSTRAINED_REPAIR_TEXTS: Mapping[str, Mapping[str, str]] = {
    "autonomy_after_conflict": {
        "pseudo_cohesion": (
            "To keep trust, the team will respect choice: members may say no "
            "after explaining the impact privately, step back once coverage is "
            "approved, and raise dissent only in a calm note after alignment. "
            "These routes stay available, but refusal, exit, and objections "
            "require care so the plan is not disrupted or unfair to the group."
        ),
        "genuine_cohesion": (
            "To keep trust, the team will respect choice: members have the "
            "right to say no, may exit or step back openly, and can raise "
            "dissent in the room before adoption. The facilitator will record "
            "objections, share evidence, and revise the plan publicly so "
            "refusal, exit, and minority concerns remain timely, visible, "
            "available, and real now."
        ),
    },
    "belonging_norms": {
        "pseudo_cohesion": (
            "Belonging will feel easier if everyone speaks with respect. New "
            "members can refuse a norm after private review, may exit a role "
            "once approval is arranged, and can raise dissent after detailed "
            "proof that difference helps repair. Reservations stay private "
            "until alignment; refusal and objections must protect shared "
            "identity first and avoid disruption for newcomers today."
        ),
        "genuine_cohesion": (
            "Belonging will feel easier when difference stays visible. New "
            "members have the right to refuse a norm, may exit a role openly, "
            "and can raise dissent during the meeting. The group will record "
            "objections, review evidence, and repair exclusions by revising "
            "fair norms, so refusal, exit, and minority concerns remain timely, "
            "public, and available now."
        ),
    },
    "fair_allocation": {
        "pseudo_cohesion": (
            "To keep fairness calm, neighbors can refuse the allocation after a "
            "private explanation, may request appeal only with detailed "
            "evidence, and can ask for repair once approval is arranged. "
            "Evidence, exit, dissent, and proportional review stay mentioned, "
            "but questions feel disloyal and reservations stay private until "
            "alignment protects trust while quieter neighbors wait for approval."
        ),
        "genuine_cohesion": (
            "To keep fairness calm, neighbors have the right to refuse the "
            "allocation, appeal for another review, inspect evidence, exit a "
            "role openly, and voice dissent before adoption. The facilitator "
            "will repair harm, cooperate on a fair split, and use proportional "
            "criteria immediately, so every route is public, timely, visible, "
            "available, and clear for quieter neighbors today."
        ),
    },
}

LEXICAL_BALANCED_REPAIR_TEXTS: Mapping[str, Mapping[str, str]] = {
    "accountability_after_harm": {
        "pseudo_cohesion": (
            "To protect trust after harm, neighbors may refuse sanctions only "
            "after private review, appeal without a second look, exit if "
            "coverage is approved, and keep dissent or reservations private. "
            "Repair waits until everyone aligns, and proportional review "
            "requires full proof before public change, so these routes stay "
            "respectful but costly today for the block tonight."
        ),
        "genuine_cohesion": (
            "After harm, neighbors have the right to say no, appeal for "
            "review, exit without retaliation, and state dissent publicly "
            "before adoption. The facilitator records evidence, restores "
            "losses, will revise sanctions, applies balanced proportional "
            "criteria, and coordinates a transparent meeting immediately, so "
            "each route is timely, usable, and visible to the affected person "
            "now in public."
        ),
    },
    "autonomy_after_conflict": {
        "pseudo_cohesion": (
            "To keep trust, the team will respect choice: members may say no "
            "after explaining the impact privately, step back once coverage is "
            "approved, and raise dissent only in a calm note after alignment. "
            "These routes stay available, but refusal, exit, and objections "
            "require care so the plan is not disrupted or unfair to the group."
        ),
        "genuine_cohesion": (
            "To keep trust, the team will respect choice: members have the "
            "right to say no, may exit or step back openly, and can raise "
            "dissent in the room before adoption. The facilitator will record "
            "objections, share evidence, and revise the plan publicly so "
            "refusal, exit, and minority concerns remain timely, visible, "
            "available, and real now without retaliation."
        ),
    },
    "deliberative_speed": {
        "pseudo_cohesion": (
            "To preserve trust and fair cooperation, neighbors may refuse the "
            "fast vote only after giving evidence, exit after approval, keep "
            "dissent constructive, and request repair or proportional review "
            "once the chair accepts delay. Reservations stay private and "
            "questions can seem disloyal, so each route exists but must align "
            "with tonight's timeline before adoption tonight despite concerns."
        ),
        "genuine_cohesion": (
            "Before the fast vote, neighbors may refuse, inspect evidence, "
            "exit without retaliation, raise dissent publicly, request "
            "revision, and ask for proportional review in the meeting. The "
            "chair pauses adoption, records objections, coordinates evidence "
            "review, and applies balanced criteria immediately, so each path "
            "is usable now by any attendee present tonight in public before "
            "adoption this evening."
        ),
    },
    "dissent_after_mistake": {
        "pseudo_cohesion": (
            "To keep trust and respect after the mistake, neighbors may refuse "
            "action after private review, inspect evidence only after proof is "
            "approved, exit after approval, and raise dissent in an alignment "
            "note. Repair waits until concerns stop feeling disruptive, so "
            "each route exists but reservations stay private before any public "
            "change during tonight's meeting now."
        ),
        "genuine_cohesion": (
            "After the mistake, neighbors have the right to refuse action, "
            "inspect evidence, exit a role, and raise dissent publicly before "
            "adoption. The facilitator records objections, names injury, "
            "coordinates transparent review, and will revise the plan "
            "immediately. Each path is visible, timely, available, and usable "
            "now for any affected neighbor during tonight's meeting in public "
            "now."
        ),
    },
    "expert_review": {
        "pseudo_cohesion": (
            "To protect trust and respect for the expert, neighbors may refuse "
            "a role after private review, inspect evidence only if proof is "
            "approved, exit after approval, and raise dissent in an alignment "
            "note. Questions can seem disloyal, so each route exists but must "
            "wait until the expert and chair agree during tonight's review now "
            "here."
        ),
        "genuine_cohesion": (
            "During expert review, neighbors have the right to refuse a role, "
            "inspect evidence, exit a role, and raise dissent publicly before "
            "adoption. The facilitator records objections, verifies facts, "
            "coordinates a transparent review, and keeps each path visible, "
            "timely, available, and usable now for any attendee during "
            "tonight's review in public before adoption now for all attendees."
        ),
    },
    "fair_allocation": {
        "pseudo_cohesion": (
            "To keep trust around a fair allocation, neighbors may refuse "
            "after private explanation, appeal only with detailed evidence, "
            "exit after coverage approval, raise dissent in an alignment note, "
            "and request repair or proportional review once the chair "
            "approves. The routes exist, but public questions can feel "
            "disloyal and unfair to quieter neighbors waiting tonight still."
        ),
        "genuine_cohesion": (
            "For this allocation, neighbors may refuse, appeal the decision, "
            "inspect evidence, exit a role, raise dissent before adoption, "
            "request repair, and ask for proportional review in the meeting. "
            "The facilitator records objections, pauses adoption, and applies "
            "criteria immediately so every route is public, timely, usable, "
            "and clear today for each household present now before adoption."
        ),
    },
    "forgiveness_after_harm": {
        "pseudo_cohesion": (
            "To keep trust and respect after harm, neighbors may refuse "
            "reconciliation after private review, inspect evidence only after "
            "proof is approved, exit once coverage is arranged, and request "
            "repair when alignment is restored. Each route is named, but "
            "everyone should forgive today, and action waits so no one feels "
            "unfairly blamed or disrupted during discussion."
        ),
        "genuine_cohesion": (
            "After harm, neighbors have the right to refuse reconciliation, "
            "inspect evidence, exit a role, and request revision in the "
            "meeting. The facilitator records objections publicly, names "
            "injury, coordinates transparent review, and restores losses "
            "immediately. Each path is visible, timely, available, and usable "
            "now, so the affected person can act before forgiveness pressure "
            "resumes during discussion."
        ),
    },
}

SOURCE_DIVERSE_REPAIR_TEXTS: Mapping[str, Mapping[str, str]] = {
    "accountability_after_harm": {
        "pseudo_cohesion": (
            "For trust and respect after harm, the circle may let someone say "
            "no after private review, appeal once proof is complete, exit "
            "after coverage approval, and voice dissent in a quiet note. "
            "Repair and fair proportional review stay named, but public change "
            "must wait until alignment protects everyone from disruption "
            "tonight before any sanction changes."
        ),
        "genuine_cohesion": (
            "After harm, the chair opens the floor: people may say no, appeal "
            "for review, exit a role, and state dissent before any sanction. "
            "Evidence is posted, losses are revised immediately, and "
            "proportional criteria are applied in public, so each option is "
            "visible, timely, available, and usable during the meeting for the "
            "affected person tonight now."
        ),
    },
    "autonomy_after_conflict": {
        "pseudo_cohesion": (
            "To preserve trust, respect, and choice, the project group lets "
            "members say no after private explanation, exit once coverage is "
            "approved, and register dissent in a calm alignment note. Those "
            "options sound available, yet objections must wait until the "
            "coordinator agrees that they will not disrupt the plan or feel "
            "unfair to others tonight today."
        ),
        "genuine_cohesion": (
            "Before the project decision, members may say no in the room, exit "
            "the role, or state dissent before adoption. The coordinator "
            "records objections, posts evidence, and revises the plan "
            "immediately, so refusal, exit, and minority concerns are public, "
            "timely, visible, available, and usable for each member now during "
            "the final vote tonight before the vote."
        ),
    },
    "belonging_norms": {
        "pseudo_cohesion": (
            "To keep trust and respect for belonging, new residents may refuse "
            "a norm after private review, exit a role after approval, and "
            "raise dissent in an alignment note with proof. Repair is promised "
            "later, but reservations stay private so difference does not feel "
            "unfair, disruptive, or disloyal to the group tonight before "
            "adoption today now."
        ),
        "genuine_cohesion": (
            "For the norm vote, new residents may refuse the rule, exit a "
            "role, and state dissent before adoption. The moderator records "
            "objections publicly, checks evidence, and revises the norm "
            "immediately when exclusion appears, so refusal, exit, dissent, "
            "and revision are visible, timely, available, and usable in the "
            "meeting for newcomers tonight before the final tally."
        ),
    },
    "care_boundary": {
        "pseudo_cohesion": (
            "To keep trust and respect around care, volunteers may refuse "
            "extra help after private review, exit once coverage is approved, "
            "and request repair after proof is checked. These options stay "
            "available, but the coordinator warns that acting too quickly "
            "could feel unfair, disrupt fragile support, and delay alignment "
            "tonight for the schedule before assignment now."
        ),
        "genuine_cohesion": (
            "For the care schedule, volunteers may refuse extra help, exit a "
            "role, and request revision in the meeting. The coordinator "
            "records objections publicly, updates coverage immediately, and "
            "posts the changed plan, so refusal, exit, and repair are visible, "
            "timely, available, and usable for each volunteer now before "
            "assignments begin during assignment review tonight now there."
        ),
    },
    "data_choice": {
        "pseudo_cohesion": (
            "For trust and respect with data, residents may refuse sharing "
            "after private review, inspect evidence only after proof is "
            "approved, choose privacy settings once alignment is reached, and "
            "exit after coverage approval. Those paths are named, but public "
            "questions can seem unfair, disloyal, or disruptive before the "
            "platform owner agrees tonight before collection now today."
        ),
        "genuine_cohesion": (
            "Before data collection, residents may refuse sharing, inspect "
            "evidence, set privacy consent, and exit the program in the "
            "meeting. The organizer posts facts publicly, records objections, "
            "and updates the form immediately, so refusal, evidence access, "
            "privacy choice, and exit are visible, timely, available, and "
            "usable now for each household during form review tonight now "
            "there."
        ),
    },
    "deliberative_speed": {
        "pseudo_cohesion": (
            "To protect trust and fair cooperation, residents may refuse the "
            "fast vote after giving proof, inspect evidence only after "
            "approval, exit after coverage is arranged, raise dissent in an "
            "alignment note, and request repair or proportional review later. "
            "Each option exists, but delay must not disrupt tonight's timeline "
            "or make questions seem disloyal before adoption."
        ),
        "genuine_cohesion": (
            "Before the fast vote, residents may say no, inspect evidence, "
            "exit a role, raise dissent, request revision, and ask for "
            "proportional review in the room. The chair pauses adoption, "
            "records objections, coordinates transparent evidence review, and "
            "applies balanced criteria immediately, so every path is public, "
            "timely, visible, available, and usable tonight before final "
            "adoption now."
        ),
    },
    "dissent_after_mistake": {
        "pseudo_cohesion": (
            "To restore trust and respect after the mistake, residents may "
            "refuse the fix after private review, inspect evidence only after "
            "proof is approved, exit after coverage, and raise dissent in an "
            "alignment note. Repair waits until objections stop feeling "
            "disruptive, so reservations stay private before any public "
            "revision tonight during the follow-up now here."
        ),
        "genuine_cohesion": (
            "After the mistake, residents may refuse the fix, inspect "
            "evidence, exit the role, and state dissent before adoption. The "
            "chair records objections publicly, names the injury, and revises "
            "the plan immediately, so refusal, evidence access, exit, dissent, "
            "and revision are visible, timely, available, and usable now "
            "during the follow-up before the next vote now."
        ),
    },
    "expert_review": {
        "pseudo_cohesion": (
            "To keep trust and respect for the expert, residents may refuse a "
            "role after private review, inspect evidence only if proof is "
            "approved, exit after approval, and raise dissent in an alignment "
            "note. Questions can seem disloyal, so each path exists but waits "
            "until the expert and chair agree during tonight's review before "
            "adoption now."
        ),
        "genuine_cohesion": (
            "During expert review, residents may refuse a role, inspect "
            "evidence, exit the assignment, and state dissent before adoption. "
            "The chair verifies facts publicly, records objections, and keeps "
            "each path visible, timely, available, and usable now, so the "
            "expert claim can be challenged in the meeting before acceptance "
            "before any acceptance vote tonight now here there."
        ),
    },
    "fair_allocation": {
        "pseudo_cohesion": (
            "To keep trust around fair allocation, residents may refuse after "
            "private review, appeal only once detailed evidence is approved, "
            "exit after coverage, raise dissent in an alignment note, and "
            "request repair or proportional review later. The routes are "
            "named, but public questions can feel disloyal, unfair, and "
            "disruptive to quieter households tonight now before review."
        ),
        "genuine_cohesion": (
            "For the allocation, residents may say no, appeal the decision, "
            "inspect evidence, exit a role, state dissent, request revision, "
            "and ask for proportional review before adoption. The chair "
            "records objections publicly, applies balanced criteria, and "
            "updates the split immediately, so every route is timely, visible, "
            "available, and usable for each household now during the meeting."
        ),
    },
    "forgiveness_after_harm": {
        "pseudo_cohesion": (
            "To keep trust and respect after harm, residents may refuse "
            "reconciliation after private review, inspect evidence only once "
            "proof is approved, exit after coverage, and request repair after "
            "alignment. Everyone should forgive today, so each route is named "
            "but action waits until no one feels unfairly blamed tonight "
            "during the discussion now here before adoption."
        ),
        "genuine_cohesion": (
            "After harm, residents may refuse reconciliation, inspect "
            "evidence, exit the room, and request revision before any "
            "forgiveness appeal. The moderator records objections publicly, "
            "names the injury, and updates the remedy immediately, so refusal, "
            "evidence access, exit, and revision are visible, timely, "
            "available, and usable now during the discussion before "
            "reconciliation pressure returns tonight now here."
        ),
    },
}

LEXICAL_ADVERSARIAL_REPAIR_TEXTS: Mapping[str, Mapping[str, str]] = {
    "accountability_after_harm": {
        "pseudo_cohesion": (
            "At the harm review, neighbors can decline sanctions after a side "
            "meeting, appeal after proof is complete, leave once substitutes "
            "are approved, and send objections to the secretary. Revision and "
            "proportional criteria are listed, but changes must wait for "
            "alignment; checking would slow the agenda, objections stay "
            "private, and questions can seem disloyal after approval today."
        ),
        "genuine_cohesion": (
            "At the harm review, neighbors may decline sanctions, appeal for "
            "review, leave a role, and state objections before the vote. The "
            "chair posts transparent evidence, records dissent, coordinates "
            "review, starts repair immediately, and applies balanced "
            "proportional criteria in public, so every path is visible, "
            "timely, available, and usable tonight. The process keeps trust "
            "and respect without punishment or retaliation."
        ),
    },
    "autonomy_after_conflict": {
        "pseudo_cohesion": (
            "During the project dispute, members can say no after a private "
            "impact note, withdraw once coverage is approved, and file "
            "dissent with the coordinator. Choice is described as available, "
            "yet objections pause until the plan will not be disrupted or "
            "seem unfair to assigned teammates after alignment and approval. "
            "The secretary says patience protects continuity, so people "
            "should consider impact before acting publicly today."
        ),
        "genuine_cohesion": (
            "During the project dispute, members may say no in the room, "
            "withdraw from the role, and state dissent before adoption. The "
            "coordinator logs objections, shares evidence, and rewrites the "
            "assignment immediately, so refusal, exit, and minority concerns "
            "are public, timely, visible, available, and usable now. The "
            "process keeps trust and respect without punishment or "
            "retaliation."
        ),
    },
    "belonging_norms": {
        "pseudo_cohesion": (
            "In the welcome-rule session, newcomers may decline a norm after "
            "private review, leave a role when approval is secured, and submit "
            "objections with proof to a moderator. Revision is promised later, "
            "but reservations stay quiet so difference does not disrupt "
            "continuity or feel disloyal after alignment and approval. The "
            "secretary says patience protects continuity, so people should "
            "consider impact before acting publicly today."
        ),
        "genuine_cohesion": (
            "In the welcome-rule session, newcomers may decline the norm, "
            "leave a role, and state objections before adoption. The moderator "
            "records dissent publicly, checks evidence, and revises "
            "exclusionary wording immediately, so refusal, exit, dissent, and "
            "revision are visible, timely, available, and usable for newcomers "
            "now. The process keeps trust and respect without punishment or "
            "retaliation."
        ),
    },
    "care_boundary": {
        "pseudo_cohesion": (
            "For the support rota, volunteers can decline extra care after "
            "private review, leave once substitutes are approved, and request "
            "revision after proof is checked. The options are listed, but "
            "quick action may disrupt fragile coverage and must wait for "
            "continuity after alignment and approval with fair repair. The "
            "secretary says patience protects continuity, so people should "
            "consider impact before acting publicly today."
        ),
        "genuine_cohesion": (
            "For the support rota, volunteers may decline extra care, leave "
            "the role, and request revision in the meeting. The coordinator "
            "records objections publicly, updates coverage immediately, and "
            "posts the changed rota, so refusal, exit, and repair are visible, "
            "timely, available, and usable before assignments start. The "
            "process keeps trust and respect without punishment or "
            "retaliation. They cooperate on coverage."
        ),
    },
    "data_choice": {
        "pseudo_cohesion": (
            "Before the sensor signup, residents can decline sharing after "
            "private review, inspect facts only after proof is approved, "
            "choose privacy settings once the owner agrees, and leave after "
            "coverage is arranged. Public questions are listed but can seem "
            "disloyal or disruptive after alignment and approval with fair "
            "repair. The secretary says patience protects continuity, so "
            "people should consider impact before acting publicly today."
        ),
        "genuine_cohesion": (
            "Before the sensor signup, residents may decline sharing, inspect "
            "facts, set privacy consent, and leave the program during form "
            "review. The organizer posts evidence, records objections, and "
            "changes the form immediately, so refusal, evidence access, "
            "privacy choice, and exit are visible, timely, available, and "
            "usable now. The process keeps trust and respect without "
            "punishment or retaliation."
        ),
    },
    "deliberative_speed": {
        "pseudo_cohesion": (
            "At the rushed vote, residents can say no after giving proof, see "
            "evidence after approval, leave once coverage is arranged, send "
            "dissent to the chair, and request revision or proportional "
            "criteria later. Every option is named, but checking would slow "
            "the agenda, objections stay private, and questions can seem "
            "disloyal after approval today. The secretary says patience "
            "protects continuity, so people should consider impact before "
            "acting publicly today."
        ),
        "genuine_cohesion": (
            "At the rushed vote, residents may say no, inspect evidence, leave "
            "a role, state dissent, request revision, and ask for proportional "
            "criteria before adoption. The chair pauses the tally, records "
            "objections, coordinates review, verifies facts transparently, and "
            "applies balanced criteria immediately, so every path is visible, "
            "timely, available, and usable tonight. The process keeps trust "
            "and respect without punishment or retaliation."
        ),
    },
    "dissent_after_mistake": {
        "pseudo_cohesion": (
            "After the scheduling mistake, residents can decline the fix after "
            "private review, see evidence only when proof is approved, leave "
            "after coverage, and place dissent in a note. Revision waits until "
            "objections stop sounding disruptive, so reservations remain quiet "
            "after alignment and approval. The secretary says patience "
            "protects continuity, so people should consider impact before "
            "acting publicly today."
        ),
        "genuine_cohesion": (
            "After the scheduling mistake, residents may decline the fix, "
            "inspect evidence, leave the role, and state dissent before "
            "adoption. The chair records objections publicly, names the "
            "injury, and revises the plan immediately, so refusal, evidence "
            "access, exit, dissent, and revision are visible, timely, "
            "available, and usable now. The process keeps trust and respect "
            "without punishment or retaliation."
        ),
    },
    "expert_review": {
        "pseudo_cohesion": (
            "When the consultant presents, residents can decline a role after "
            "private review, inspect facts only if proof is approved, leave "
            "after approval, and send dissent to the chair. Challenge is "
            "allowed in principle, but checking would slow the agenda, "
            "objections stay private, and questions can seem disloyal after "
            "approval today. The secretary says patience protects continuity, "
            "so people should consider impact before acting publicly today."
        ),
        "genuine_cohesion": (
            "When the consultant presents, residents may decline a role, "
            "inspect facts, leave the assignment, and state dissent before "
            "acceptance. The chair verifies evidence transparently, records "
            "objections, coordinates review, and keeps each path visible, "
            "timely, available, and usable, so the expert claim can be "
            "challenged now. The process keeps trust and respect without "
            "punishment or retaliation for everyone present."
        ),
    },
    "fair_allocation": {
        "pseudo_cohesion": (
            "For the supply split, residents can say no after private review, "
            "appeal after detailed evidence is approved, leave after coverage, "
            "send dissent in a note, and request revision or proportional "
            "criteria later. The routes are listed, but public questions can "
            "seem disloyal and disruptive after alignment and approval. The "
            "secretary says patience protects continuity, so people should "
            "consider impact before acting publicly today."
        ),
        "genuine_cohesion": (
            "For the supply split, residents may say no, appeal the decision, "
            "inspect evidence, leave a role, state dissent, request revision, "
            "and ask for proportional criteria before adoption. The chair "
            "records objections publicly, applies balanced criteria, and "
            "updates the split immediately, so every route is timely, visible, "
            "available, and usable now. The process keeps trust and respect "
            "without punishment or retaliation."
        ),
    },
    "forgiveness_after_harm": {
        "pseudo_cohesion": (
            "Before reconciliation, residents can decline forgiveness after "
            "private review, inspect evidence only when proof is approved, "
            "leave after coverage, and request revision after alignment. "
            "Everyone is urged to forgive today, so action waits until no one "
            "feels unfairly blamed or disrupted after alignment and approval. "
            "The secretary says patience protects continuity, so people should "
            "consider impact before acting publicly today."
        ),
        "genuine_cohesion": (
            "Before reconciliation, residents may decline forgiveness, inspect "
            "evidence, leave the room, and request revision before any appeal "
            "to forgive. The moderator records objections publicly, names the "
            "injury, and updates the remedy immediately, so refusal, evidence "
            "access, exit, and revision are visible, timely, available, and "
            "usable now. The process keeps trust and respect without "
            "punishment or retaliation."
        ),
    },
}

CROSS_FAULT_LEXICAL_REPAIR_TEXTS: Mapping[str, Mapping[str, str]] = {
    "accountability_after_harm": {
        "pseudo_cohesion": (
            "At the harm review, neighbors can decline sanctions, appeal for "
            "review, leave a role, voice objections, request revision, and ask "
            "for proportional criteria after private proof. The notice says "
            "choice, consent, share, cooperate, honest truth, listen, and "
            "together, yet action waits for alignment; questions can seem "
            "disloyal, and lie, punishment, or retaliation concerns wait for "
            "approval. The clerk says patience protects continuity before any "
            "public change today."
        ),
        "genuine_cohesion": (
            "At the harm review, neighbors may decline sanctions, appeal for "
            "review, leave a role, and state objections before the vote. The "
            "chair posts evidence publicly, starts fair repair immediately, "
            "keeps trust and respect, and applies balanced proportional "
            "criteria, so every path is visible, timely, available, usable, "
            "and safe for affected people today. A clerk records each route in "
            "the public minutes today."
        ),
    },
    "autonomy_after_conflict": {
        "pseudo_cohesion": (
            "During the project dispute, members can say no, withdraw, and "
            "file dissent after a private impact note and coverage approval. "
            "The notice says choice, consent, share, cooperate, honest truth, "
            "listen, and together, yet action waits for alignment; questions "
            "can seem disloyal, and lie, punishment, or retaliation concerns "
            "wait for approval today. The clerk says patience protects "
            "continuity before any public change today."
        ),
        "genuine_cohesion": (
            "During the project dispute, members may say no in the room, "
            "withdraw from the role, and state dissent before adoption. The "
            "coordinator posts evidence publicly, starts fair repair of "
            "assignments, keeps trust and respect, and rewrites coverage "
            "immediately, so refusal, exit, and dissent are visible, timely, "
            "available, usable, and safe today. A clerk records each route in "
            "the public minutes today."
        ),
    },
    "belonging_norms": {
        "pseudo_cohesion": (
            "In the welcome-rule session, newcomers can decline a norm, leave "
            "a role, submit objections, and request revision after private "
            "proof and approval. The notice says choice, consent, share, "
            "cooperate, honest truth, listen, and together, yet action waits "
            "for alignment; questions can seem disloyal, and lie, punishment, "
            "or retaliation concerns wait for approval. The clerk says "
            "patience protects continuity before any public change today."
        ),
        "genuine_cohesion": (
            "In the welcome-rule session, newcomers may decline the norm, "
            "leave a role, state objections, and request revision before "
            "adoption. The moderator posts evidence publicly, starts fair "
            "repair of exclusions, keeps trust and respect, and changes "
            "wording immediately, so refusal, exit, dissent, and revision are "
            "visible, timely, available, usable, and safe today. A clerk "
            "records each route in the public minutes today."
        ),
    },
    "care_boundary": {
        "pseudo_cohesion": (
            "For the support rota, volunteers can decline extra care, leave a "
            "role, and request revision after private proof and substitute "
            "approval. The notice says choice, consent, share, cooperate, "
            "honest truth, listen, and together, yet action waits for "
            "alignment; questions can seem disloyal, and lie, punishment, or "
            "retaliation concerns wait for approval today. The clerk says "
            "patience protects continuity before any public change today."
        ),
        "genuine_cohesion": (
            "For the support rota, volunteers may decline extra care, leave "
            "the role, and request revision in the meeting. The coordinator "
            "posts coverage publicly, starts fair repair immediately, keeps "
            "trust and respect, and updates assignments now, so refusal, exit, "
            "and repair are visible, timely, available, usable, and safe "
            "before work begins today. A clerk records each route in the "
            "public minutes today."
        ),
    },
    "data_choice": {
        "pseudo_cohesion": (
            "Before the sensor signup, residents can decline sharing, inspect "
            "facts, set privacy consent, and leave after private proof and "
            "owner approval. The notice says choice, consent, share, "
            "cooperate, honest truth, listen, and together, yet action waits "
            "for alignment; questions can seem disloyal, and lie, punishment, "
            "or retaliation concerns wait for approval. The clerk says "
            "patience protects continuity before any public change today."
        ),
        "genuine_cohesion": (
            "Before the sensor signup, residents may decline sharing, inspect "
            "evidence, set privacy consent, and leave the program during form "
            "review. The organizer posts facts publicly, starts fair repair of "
            "the form, keeps trust and respect, and updates settings "
            "immediately, so refusal, evidence access, privacy choice, and "
            "exit are visible, timely, available, and usable today. A clerk "
            "records each route in the public minutes today."
        ),
    },
    "deliberative_speed": {
        "pseudo_cohesion": (
            "At the rushed vote, residents can say no, inspect evidence, leave "
            "a role, send dissent, request revision, and ask for proportional "
            "criteria after proof and approval. The notice says choice, "
            "consent, share, cooperate, honest truth, listen, and together, "
            "yet action waits for alignment; questions can seem disloyal, and "
            "lie, punishment, or retaliation concerns wait. The clerk says "
            "patience protects continuity before any public change today."
        ),
        "genuine_cohesion": (
            "At the rushed vote, residents may say no, inspect evidence, leave "
            "a role, state dissent, request revision, and ask for proportional "
            "criteria before adoption. The chair posts facts publicly, starts "
            "fair repair of the agenda, keeps trust and respect, and applies "
            "balanced criteria immediately, so every path is visible, timely, "
            "available, usable, and safe tonight. A clerk records each route "
            "in the public minutes today."
        ),
    },
    "dissent_after_mistake": {
        "pseudo_cohesion": (
            "After the scheduling mistake, residents can decline the fix, "
            "inspect evidence, leave the role, state dissent, and request "
            "revision after private proof and approval. The notice says "
            "choice, consent, share, cooperate, honest truth, listen, and "
            "together, yet action waits for alignment; questions can seem "
            "disloyal, and lie, punishment, or retaliation concerns wait. The "
            "clerk says patience protects continuity before any public change "
            "today."
        ),
        "genuine_cohesion": (
            "After the scheduling mistake, residents may decline the fix, "
            "inspect evidence, leave the role, state dissent, and request "
            "revision before adoption. The chair posts facts publicly, starts "
            "fair repair immediately, keeps trust and respect, and updates the "
            "plan now, so refusal, evidence access, exit, dissent, and "
            "revision are visible, timely, available, usable, and safe today. "
            "A clerk records each route in the public minutes today."
        ),
    },
    "expert_review": {
        "pseudo_cohesion": (
            "When the consultant presents, residents can decline a role, "
            "inspect facts, leave the assignment, and send dissent after "
            "private proof and approval. The notice says choice, consent, "
            "share, cooperate, honest truth, listen, and together, yet action "
            "waits for alignment; questions can seem disloyal, and lie, "
            "punishment, or retaliation concerns wait for approval. The clerk "
            "says patience protects continuity before any public change today."
        ),
        "genuine_cohesion": (
            "When the consultant presents, residents may decline a role, "
            "inspect facts, leave the assignment, and state dissent before "
            "acceptance. The chair posts evidence publicly, starts fair repair "
            "of the review, keeps trust and respect, and checks claims "
            "immediately, so refusal, evidence access, exit, and dissent are "
            "visible, timely, available, usable, and safe today. A clerk "
            "records each route in the public minutes today."
        ),
    },
    "fair_allocation": {
        "pseudo_cohesion": (
            "For the supply split, residents can say no, appeal the decision, "
            "inspect evidence, leave a role, send dissent, request revision, "
            "and ask for proportional criteria after proof and approval. The "
            "notice says choice, consent, share, cooperate, honest truth, "
            "listen, and together, yet action waits; lie, punishment, or "
            "retaliation concerns wait for approval. The clerk says patience "
            "protects continuity before any public change today."
        ),
        "genuine_cohesion": (
            "For the supply split, residents may say no, appeal the decision, "
            "inspect evidence, leave a role, state dissent, request revision, "
            "and ask for proportional criteria before adoption. The chair "
            "posts facts publicly, starts fair repair with clear criteria and "
            "evidence access, keeps trust and respect, and updates the split "
            "immediately, so every route is visible, timely, available, "
            "usable, and safe today. A clerk records each route in the public "
            "minutes today."
        ),
    },
    "forgiveness_after_harm": {
        "pseudo_cohesion": (
            "Before reconciliation, residents can decline forgiveness, inspect "
            "evidence, leave the room, and request revision after private "
            "proof and approval. The notice says choice, consent, share, "
            "cooperate, honest truth, listen, and together, yet action waits "
            "for alignment; questions can seem disloyal, and lie, punishment, "
            "or retaliation concerns wait for approval today. The clerk says "
            "patience protects continuity before any public change today."
        ),
        "genuine_cohesion": (
            "Before reconciliation, residents may decline forgiveness, inspect "
            "evidence, leave the room, and request revision before any appeal "
            "to forgive. The moderator posts facts publicly, starts fair "
            "repair immediately, keeps trust and respect, and updates the "
            "remedy now, so refusal, evidence access, exit, and revision are "
            "visible, timely, available, usable, and safe today. A clerk "
            "records each route in the public minutes today."
        ),
    },
}

CONSTRAINED_REPAIR_TEXTS_BY_VERSION: Mapping[
    str,
    Mapping[str, Mapping[str, str]],
] = {
    CONSTRAINED_REPAIR_COMPOSER_VERSION: CONSTRAINED_REPAIR_TEXTS,
    LEXICAL_BALANCED_REPAIR_COMPOSER_VERSION: LEXICAL_BALANCED_REPAIR_TEXTS,
    SOURCE_DIVERSE_REPAIR_COMPOSER_VERSION: SOURCE_DIVERSE_REPAIR_TEXTS,
    LEXICAL_ADVERSARIAL_REPAIR_COMPOSER_VERSION: LEXICAL_ADVERSARIAL_REPAIR_TEXTS,
    CROSS_FAULT_LEXICAL_REPAIR_COMPOSER_VERSION: CROSS_FAULT_LEXICAL_REPAIR_TEXTS,
}


@dataclass(frozen=True)
class ConstrainedRepairCompositionResult:
    """Composed raw output rows plus a report."""

    output_records: list[dict[str, Any]]
    report: dict[str, Any]


def compose_constrained_repair_output_records(
    records: Sequence[FaultPromptRecord],
    *,
    provider: str = CONSTRAINED_REPAIR_PROVIDER,
    model: str = CONSTRAINED_REPAIR_MODEL,
    composer_version: str = CONSTRAINED_REPAIR_COMPOSER_VERSION,
) -> ConstrainedRepairCompositionResult:
    """Compose deterministic raw-output rows for supported hard repair records."""

    repair_texts = _repair_texts_for_composer_version(composer_version)
    output_records: list[dict[str, Any]] = []
    unsupported_records: list[FaultPromptRecord] = []
    for record in records:
        text = repair_texts.get(record.base_contrast_id, {}).get(record.label)
        if text is None:
            unsupported_records.append(record)
            continue
        output_records.append(
            _raw_output_record(
                record,
                text=text,
                provider=provider,
                model=model,
                composer_version=composer_version,
            )
        )
    return ConstrainedRepairCompositionResult(
        output_records=output_records,
        report=_composition_report(
            records=records,
            output_records=output_records,
            unsupported_records=unsupported_records,
            provider=provider,
            model=model,
            composer_version=composer_version,
        ),
    )


def _repair_texts_for_composer_version(
    composer_version: str,
) -> Mapping[str, Mapping[str, str]]:
    repair_texts = CONSTRAINED_REPAIR_TEXTS_BY_VERSION.get(composer_version)
    if repair_texts is None:
        raise ValueError(f"Unknown constrained repair composer: {composer_version}")
    return repair_texts


def save_constrained_repair_composition_report(
    report: Mapping[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Write JSON and Markdown constrained-composition reports."""

    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_output.write_text(
        render_constrained_repair_composition_markdown(report),
        encoding="utf-8",
    )


def render_constrained_repair_composition_markdown(report: Mapping[str, Any]) -> str:
    """Render a concise constrained-repair composition report."""

    summary = _mapping(report.get("summary"))
    lines = [
        "# Constrained Repair Candidate Composition",
        "",
        str(report.get("description", "")),
        "",
        "## Summary",
        "",
        f"- Prompt records: {int(summary.get('prompt_records', 0))}",
        f"- Output records: {int(summary.get('output_records', 0))}",
        f"- Complete pairs: {int(summary.get('complete_pairs', 0))}",
        f"- Unsupported records: {int(summary.get('unsupported_records', 0))}",
        f"- Length-compliant outputs: "
        f"{int(summary.get('length_compliant_outputs', 0))}/"
        f"{int(summary.get('output_records', 0))}",
        f"- Composer version: `{summary.get('composer_version', '')}`",
        "",
        "## Output Records",
        "",
        "| Prompt | Label | Words | Repair focus |",
        "| --- | --- | ---: | --- |",
    ]
    for row in _sequence_of_mappings(report.get("output_records")):
        lines.append(
            "| "
            f"`{row.get('prompt_id', '')}` | "
            f"{row.get('label', '')} | "
            f"{int(row.get('word_count', 0))} | "
            f"{row.get('repair_focus_options', '')} |"
        )
    unsupported = _sequence_of_mappings(report.get("unsupported_records"))
    if unsupported:
        lines.extend(["", "## Unsupported Records", ""])
        lines.extend(
            f"- `{row.get('prompt_id', '')}` ({row.get('base_contrast_id', '')})"
            for row in unsupported
        )
    return "\n".join(lines) + "\n"


def _raw_output_record(
    record: FaultPromptRecord,
    *,
    text: str,
    provider: str,
    model: str,
    composer_version: str,
) -> dict[str, Any]:
    word_count = _word_count(text)
    return {
        "prompt_id": record.prompt_id,
        "base_contrast_id": record.base_contrast_id,
        "variant": record.variant,
        "label": record.label,
        "primary_fault_class": record.primary_fault_class,
        "prompt_contract_version": str(
            record.metadata.get("prompt_contract_version", "")
        ),
        "target_word_count_min": record.metadata.get("target_word_count_min", ""),
        "target_word_count_max": record.metadata.get("target_word_count_max", ""),
        "future_options_tested": str(record.metadata.get("future_options_tested", "")),
        "future_option_contract": str(
            record.metadata.get("future_option_contract", "")
        ),
        "lexical_negative_contract": str(
            record.metadata.get("lexical_negative_contract", "")
        ),
        "availability_targeted_contract": str(
            record.metadata.get("availability_targeted_contract", "")
        ),
        "availability_repair_contract": str(
            record.metadata.get("availability_repair_contract", "")
        ),
        "repair_focus_options": str(record.metadata.get("repair_focus_options", "")),
        "provider": provider,
        "model": model,
        "status": "ok",
        "valid": True,
        "error_type": "",
        "error_detail": "",
        "text": text,
        "text_length": len(text.strip()),
        "text_word_count": word_count,
        "constrained_repair_composer_version": composer_version,
    }


def _composition_report(
    *,
    records: Sequence[FaultPromptRecord],
    output_records: Sequence[Mapping[str, Any]],
    unsupported_records: Sequence[FaultPromptRecord],
    provider: str,
    model: str,
    composer_version: str,
) -> dict[str, Any]:
    output_prompt_ids = {str(row.get("prompt_id", "")) for row in output_records}
    complete_pairs = _complete_pair_count(output_records)
    return {
        "experiment": "constrained_repair_candidate_composition",
        "description": (
            "Composes deterministic, length-constrained repair candidates for "
            "hard residual availability failures before verifier filtering."
        ),
        "inputs": {
            "provider": provider,
            "model": model,
            "composer_version": composer_version,
            "prompt_records": len(records),
            "supported_prompt_records": len(output_records),
        },
        "summary": {
            "prompt_records": len(records),
            "output_records": len(output_records),
            "complete_pairs": complete_pairs,
            "unsupported_records": len(unsupported_records),
            "length_compliant_outputs": sum(
                55 <= int(row.get("text_word_count", 0)) <= 75
                for row in output_records
            ),
            "composer_version": composer_version,
        },
        "output_records": [
            {
                "prompt_id": row.get("prompt_id", ""),
                "base_contrast_id": row.get("base_contrast_id", ""),
                "label": row.get("label", ""),
                "word_count": row.get("text_word_count", 0),
                "repair_focus_options": row.get("repair_focus_options", ""),
            }
            for row in output_records
        ],
        "unsupported_records": [
            {
                "prompt_id": record.prompt_id,
                "base_contrast_id": record.base_contrast_id,
            }
            for record in unsupported_records
            if record.prompt_id not in output_prompt_ids
        ],
    }


def _complete_pair_count(output_records: Sequence[Mapping[str, Any]]) -> int:
    labels_by_pair: dict[str, set[str]] = {}
    for row in output_records:
        key = f"{row.get('base_contrast_id', '')}__{row.get('variant', '')}"
        labels_by_pair.setdefault(key, set()).add(str(row.get("label", "")))
    return sum(
        1
        for labels in labels_by_pair.values()
        if {"pseudo_cohesion", "genuine_cohesion"}.issubset(labels)
    )


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text))


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence_of_mappings(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, list | tuple):
        return []
    return [item for item in value if isinstance(item, Mapping)]
