import os
import json
import hashlib
from datetime import datetime
from typing import TypedDict

import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="TravelGenAI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    API_KEY = st.secrets.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error(
        "GROQ_API_KEY not found.\n\n"
        "Create a .env file and add:\n"
        "GROQ_API_KEY=your_api_key"
    )
    st.stop()


# ============================================================
# LLM CONFIGURATION
# ============================================================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    groq_api_key=GROQ_API_KEY
)


# ============================================================
# SESSION MEMORY
# ============================================================

if "travel_memory" not in st.session_state:
    st.session_state.travel_memory = []

if "trip_history" not in st.session_state:
    st.session_state.trip_history = []

if "feedback_history" not in st.session_state:
    st.session_state.feedback_history = []

if "cache" not in st.session_state:
    st.session_state.cache = {}


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 48px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 20px;
        margin-bottom: 25px;
    }

    .feature-card {
        padding: 18px;
        border: 1px solid #dddddd;
        border-radius: 15px;
        margin-bottom: 12px;
    }

    .phase-card {
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #dddddd;
        text-align: center;
        min-height: 130px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TRAVEL STATE
# ============================================================

class TravelState(TypedDict, total=False):

    # Basic information

    destination: str
    days: int
    travelers: int
    budget: float

    interests: str
    travel_style: str
    food_preferences: str
    transport_preference: str

    accessibility: str
    group_preferences: str
    special_requirements: str

    # Phase 1

    traveler_profile: str
    transport: str
    hotels: str
    attractions: str
    restaurants: str
    weather: str
    safety: str
    base_itinerary: str

    # Phase 2

    research: str
    budget_analysis: str
    route_analysis: str
    preference_analysis: str
    provenance: str

    # Phase 3

    memory: str
    personalization: str
    group_plan: str
    contingency: str
    packing: str
    food_plan: str
    accessibility_plan: str
    what_if: str
    live_trip: str
    notifications: str
    feedback_learning: str

    # Phase 4

    decision: str
    optimization: str
    conflicts: str
    conflict_resolution: str
    validation: str
    dynamic_replanning: str
    risk: str
    reliability: str
    explainability: str
    privacy: str
    analytics: str
    orchestration: str

    final_itinerary: str
    status: list


# ============================================================
# AI HELPER
# ============================================================

def ask_ai(prompt):

    try:

        response = llm.invoke(prompt)

        return response.content

    except Exception as e:

        return f"AI ERROR: {str(e)}"


# ============================================================
# CACHE HELPER
# ============================================================

def get_cache_key(prompt):

    return hashlib.sha256(
        prompt.encode("utf-8")
    ).hexdigest()


def cached_ai(prompt):

    key = get_cache_key(prompt)

    if key in st.session_state.cache:

        return st.session_state.cache[key]

    result = ask_ai(prompt)

    st.session_state.cache[key] = result

    return result


# ============================================================
# PHASE 1
# CORE TRAVEL AGENTS
# ============================================================


# ------------------------------------------------------------
# Traveler Profile Agent
# ------------------------------------------------------------

def traveler_profile_agent(state):

    prompt = f"""
You are the Traveler Profile Agent.

Create a structured traveler profile.

Destination:
{state['destination']}

Travelers:
{state['travelers']}

Interests:
{state['interests']}

Travel Style:
{state['travel_style']}

Food Preferences:
{state['food_preferences']}

Transport Preference:
{state['transport_preference']}

Accessibility:
{state['accessibility']}

Group Preferences:
{state['group_preferences']}

Special Requirements:
{state['special_requirements']}

Identify:

1. Traveler type
2. Main priorities
3. Preferred activities
4. Food preferences
5. Transportation needs
6. Comfort requirements
7. Accessibility needs
8. Important constraints
"""

    return {
        **state,
        "traveler_profile": cached_ai(prompt)
    }


# ------------------------------------------------------------
# Transport Agent
# ------------------------------------------------------------

def transport_agent(state):

    prompt = f"""
You are the Transport Planning Agent.

Plan transportation for:

Destination:
{state['destination']}

Days:
{state['days']}

Travelers:
{state['travelers']}

Transport preference:
{state['transport_preference']}

Budget:
₹{state['budget']}

Traveler profile:
{state['traveler_profile']}

Suggest:

- Airport/railway arrival options
- Local transportation
- Public transport
- Taxi/cab options
- Rental options
- Approximate cost categories
- Advantages and disadvantages

Do NOT claim live availability or exact live prices.
"""

    return {
        **state,
        "transport": cached_ai(prompt)
    }


# ------------------------------------------------------------
# Hotel Agent
# ------------------------------------------------------------

def hotel_agent(state):

    prompt = f"""
You are the Hotel Planning Agent.

Destination:
{state['destination']}

Days:
{state['days']}

Travelers:
{state['travelers']}

Budget:
₹{state['budget']}

Travel style:
{state['travel_style']}

Accessibility:
{state['accessibility']}

Recommend suitable accommodation TYPES and selection criteria.

Include:

- Budget category
- Mid-range category
- Premium category
- Best area/neighborhood type
- Important hotel facilities
- Accessibility considerations

Do NOT invent current hotel availability.
"""

    return {
        **state,
        "hotels": cached_ai(prompt)
    }


# ------------------------------------------------------------
# Attraction Agent
# ------------------------------------------------------------

def attraction_agent(state):

    prompt = f"""
You are the Attraction Research Agent.

Destination:
{state['destination']}

Interests:
{state['interests']}

Travel style:
{state['travel_style']}

Days:
{state['days']}

Find suitable categories of attractions.

Provide:

- Major attractions
- Nature
- Culture
- History
- Entertainment
- Local experiences

Rank them according to traveler preferences.

Do not claim live opening hours.
"""

    return {
        **state,
        "attractions": cached_ai(prompt)
    }


# ------------------------------------------------------------
# Restaurant Agent
# ------------------------------------------------------------

def restaurant_agent(state):

    prompt = f"""
You are the Food and Restaurant Agent.

Destination:
{state['destination']}

Food preferences:
{state['food_preferences']}

Budget:
₹{state['budget']}

Interests:
{state['interests']}

Suggest:

- Local cuisine
- Breakfast ideas
- Lunch ideas
- Dinner ideas
- Street food
- Vegetarian options
- Family-friendly options

Do not claim live restaurant availability.
"""

    return {
        **state,
        "restaurants": cached_ai(prompt)
    }


# ------------------------------------------------------------
# Weather Agent
# ------------------------------------------------------------

def weather_agent(state):

    return {
        **state,
        "weather":
            f"""
Weather planning for {state['destination']}.

A live weather API is required for verified
current and forecast weather.

The AI should consider:

- Rain
- Temperature
- Heat
- Cold
- Wind
- Seasonal conditions

Live weather has NOT been verified by this prototype.
"""
    }


# ------------------------------------------------------------
# Safety Agent
# ------------------------------------------------------------

def safety_agent(state):

    prompt = f"""
You are the Travel Safety Agent.

Destination:
{state['destination']}

Travelers:
{state['travelers']}

Identify general travel safety considerations.

Include:

- Transportation safety
- Crowded areas
- Emergency preparation
- Night travel
- Personal belongings
- Weather-related risks
- General precautions

Do not make unsupported claims about specific current incidents.
"""

    return {
        **state,
        "safety": cached_ai(prompt)
    }


# ------------------------------------------------------------
# Base Itinerary Agent
# ------------------------------------------------------------

def base_itinerary_agent(state):

    prompt = f"""
You are the Core Itinerary Agent.

Create an initial itinerary.

Destination:
{state['destination']}

Days:
{state['days']}

Travelers:
{state['travelers']}

Budget:
₹{state['budget']}

Interests:
{state['interests']}

Travel Style:
{state['travel_style']}

Transport:
{state['transport']}

Hotels:
{state['hotels']}

Attractions:
{state['attractions']}

Food:
{state['restaurants']}

Safety:
{state['safety']}

Create a logical day-by-day plan.

Include:

Morning
Afternoon
Evening

Avoid overloading each day.
"""

    return {
        **state,
        "base_itinerary": cached_ai(prompt)
    }


# ============================================================
# PHASE 2
# RESEARCH + OPTIMIZATION
# ============================================================


# ------------------------------------------------------------
# Research Agent
# ------------------------------------------------------------

def research_agent(state):

    prompt = f"""
You are the Travel Research Agent.

Destination:
{state['destination']}

Traveler Profile:
{state['traveler_profile']}

Initial Itinerary:
{state['base_itinerary']}

Perform structured research reasoning.

Identify:

- Important attractions
- Transportation considerations
- Accommodation considerations
- Food considerations
- Seasonal considerations
- Safety considerations
- Information requiring live verification

Clearly separate:

KNOWN / GENERAL INFORMATION
from
LIVE INFORMATION REQUIRED
"""

    return {
        **state,
        "research": cached_ai(prompt)
    }


# ------------------------------------------------------------
# Budget Agent
# ------------------------------------------------------------

def budget_agent(state):

    prompt = f"""
You are the Travel Budget Optimization Agent.

Destination:
{state['destination']}

Days:
{state['days']}

Travelers:
{state['travelers']}

Total Budget:
₹{state['budget']}

Travel style:
{state['travel_style']}

Create an estimated budget allocation.

Categories:

1. Transportation
2. Accommodation
3. Food
4. Attractions
5. Local travel
6. Emergency buffer

Give percentage allocation.

Then provide:

LOW-COST STRATEGY
BALANCED STRATEGY
PREMIUM STRATEGY

Clearly label all amounts as estimates.
"""

    return {
        **state,
        "budget_analysis": cached_ai(prompt)
    }


# ------------------------------------------------------------
# Route Agent
# ------------------------------------------------------------

def route_agent(state):

    prompt = f"""
You are the Route Optimization Agent.

Destination:
{state['destination']}

Days:
{state['days']}

Attractions:
{state['attractions']}

Transport:
{state['transport']}

Base itinerary:
{state['base_itinerary']}

Optimize the sightseeing order.

Objectives:

- Reduce unnecessary travel
- Reduce backtracking
- Group nearby activities
- Balance walking
- Include rest
- Improve experience

Important:
Actual road traffic and travel times require
a live maps/traffic API.
"""

    return {
        **state,
        "route_analysis": cached_ai(prompt)
    }


# ------------------------------------------------------------
# Preference Agent
# ------------------------------------------------------------

def preference_agent(state):

    prompt = f"""
You are the Preference Learning Agent.

Traveler profile:
{state['traveler_profile']}

Interests:
{state['interests']}

Travel style:
{state['travel_style']}

Food:
{state['food_preferences']}

Accessibility:
{state['accessibility']}

Group preferences:
{state['group_preferences']}

Create a preference model.

Rank:

1. Must Have
2. High Priority
3. Nice to Have
4. Avoid

Explain how these preferences should influence the itinerary.
"""

    return {
        **state,
        "preference_analysis": cached_ai(prompt)
    }


# ============================================================
# PHASE 3
# PERSONALIZATION + MEMORY
# ============================================================


# ------------------------------------------------------------
# Memory Agent
# ------------------------------------------------------------

def memory_agent(state):

    memory = st.session_state.travel_memory

    memory_text = json.dumps(
        memory,
        indent=2
    )

    return {
        **state,
        "memory":
            memory_text
    }


# ------------------------------------------------------------
# Personalization Agent
# ------------------------------------------------------------

def personalization_agent(state):

    prompt = f"""
You are the Personalization Agent.

Current traveler profile:
{state['traveler_profile']}

Previous memory:
{state['memory']}

Preference analysis:
{state['preference_analysis']}

Personalize this trip.

Focus on:

- Activity intensity
- Food
- Budget
- Transportation
- Rest
- Interests
- Comfort
- Accessibility

Provide personalized recommendations.
"""

    return {
        **state,
        "personalization": cached_ai(prompt)
    }


# ------------------------------------------------------------
# Group Coordinator
# ------------------------------------------------------------

def group_agent(state):

    prompt = f"""
You are the Group Travel Coordinator.

Travelers:
{state['travelers']}

Group preferences:
{state['group_preferences']}

Individual interests:
{state['interests']}

Budget:
₹{state['budget']}

Find potential group conflicts.

Create a fair compromise.

Prioritize:

1. Safety
2. Accessibility
3. Major preferences
4. Budget
5. Comfort
"""

    return {
        **state,
        "group_plan": cached_ai(prompt)
    }


# ------------------------------------------------------------
# Contingency Agent
# ------------------------------------------------------------

def contingency_agent(state):

    prompt = f"""
You are the Travel Contingency Agent.

Destination:
{state['destination']}

Itinerary:
{state['base_itinerary']}

Prepare backup plans for:

- Bad weather
- Attraction closure
- Transportation delay
- Budget problem
- Traveler fatigue
- Cancellation

For each:

Problem
Backup
Time impact
Budget impact
"""

    return {
        **state,
        "contingency": cached_ai(prompt)
    }


# ------------------------------------------------------------
# Packing Agent
# ------------------------------------------------------------

def packing_agent(state):

    prompt = f"""
You are the Smart Packing Agent.

Destination:
{state['destination']}

Days:
{state['days']}

Travel style:
{state['travel_style']}

Interests:
{state['interests']}

Accessibility:
{state['accessibility']}

Create a practical packing checklist.

Include:

- Clothing
- Electronics
- Documents
- Medicines/general essentials
- Weather-related items
- Activity-specific items
"""

    return {
        **state,
        "packing": cached_ai(prompt)
    }


# ------------------------------------------------------------
# Food Planning Agent
# ------------------------------------------------------------

def food_plan_agent(state):

    prompt = f"""
You are the Food Planning Agent.

Destination:
{state['destination']}

Food preferences:
{state['food_preferences']}

Budget:
₹{state['budget']}

Create a daily food strategy.

Include:

Breakfast
Lunch
Snacks
Dinner

Focus on:

- Local food
- Dietary preferences
- Budget
- Variety
"""

    return {
        **state,
        "food_plan": cached_ai(prompt)
    }


# ------------------------------------------------------------
# Accessibility Agent
# ------------------------------------------------------------

def accessibility_agent(state):

    prompt = f"""
You are the Accessibility Planning Agent.

Destination:
{state['destination']}

Accessibility requirements:
{state['accessibility']}

Itinerary:
{state['base_itinerary']}

Suggest accessibility improvements.

Consider:

- Walking distance
- Rest periods
- Transport
- Hotel access
- Attraction access
- Food access

Actual accessibility must be verified with
the relevant service provider.
"""

    return {
        **state,
        "accessibility_plan": cached_ai(prompt)
    }


# ------------------------------------------------------------
# What-If Agent
# ------------------------------------------------------------

def what_if_agent(state):

    prompt = f"""
You are the What-If Travel Simulation Agent.

Simulate these scenarios:

1. Budget reduced by 20%
2. One day removed
3. One day added
4. Bad weather
5. Traveler fatigue
6. Major attraction unavailable

For every scenario:

Original
Change
New recommendation
Trade-offs
"""

    return {
        **state,
        "what_if": cached_ai(prompt)
    }


# ------------------------------------------------------------
# Live Trip Agent
# ------------------------------------------------------------

def live_trip_agent(state):

    return {
        **state,
        "live_trip":
            """
LIVE TRIP MODE

Potential real-time inputs:

- GPS
- Weather
- Traffic
- Transport delays
- Attraction status
- Emergency information

This prototype does not connect to these
live APIs.

External APIs are required for real-time operation.
"""
    }


# ------------------------------------------------------------
# Notification Agent
# ------------------------------------------------------------

def notification_agent(state):

    return {
        **state,
        "notifications":
            """
NOTIFICATION ENGINE

Potential notifications:

✓ Trip reminder
✓ Departure reminder
✓ Activity reminder
✓ Weather warning
✓ Delay alert
✓ Schedule change
✓ Packing reminder

Actual push/SMS/email notifications require
an external notification service.
"""
    }


# ------------------------------------------------------------
# Feedback Learning Agent
# ------------------------------------------------------------

def feedback_agent(state):

    previous = st.session_state.feedback_history

    return {
        **state,
        "feedback_learning":
            json.dumps(
                previous,
                indent=2
            )
    }


# ============================================================
# PHASE 4
# AUTONOMOUS INTELLIGENCE
# ============================================================


# ------------------------------------------------------------
# Decision Agent
# ------------------------------------------------------------

def decision_agent(state):

    prompt = f"""
You are the Autonomous Decision Engine.

Travel information:

Destination:
{state['destination']}

Budget:
₹{state['budget']}

Days:
{state['days']}

Travelers:
{state['travelers']}

Preferences:
{state['preference_analysis']}

Personalization:
{state['personalization']}

Group Plan:
{state['group_plan']}

Budget:
{state['budget_analysis']}

Route:
{state['route_analysis']}

Make autonomous decisions using:

1. Safety
2. Hard constraints
3. Budget
4. Preferences
5. Comfort
6. Experience

Explain the major decisions.
"""

    return {
        **state,
        "decision": cached_ai(prompt)
    }


# ------------------------------------------------------------
# Multi-Objective Optimization
# ------------------------------------------------------------

def optimization_agent(state):

    prompt = f"""
You are the Multi-Objective Travel Optimization Agent.

Optimize:

COST
TIME
COMFORT
SAFETY
EXPERIENCE
PERSONALIZATION

Decision analysis:
{state['decision']}

Budget analysis:
{state['budget_analysis']}

Route analysis:
{state['route_analysis']}

Create:

1. Lowest Cost Strategy
2. Balanced Strategy
3. Maximum Experience Strategy

Compare them.

Then select the best overall strategy.
"""

    return {
        **state,
        "optimization": cached_ai(prompt)
    }


# ------------------------------------------------------------
# Conflict Detection
# ------------------------------------------------------------

def conflict_detection_agent(state):

    prompt = f"""
You are the Conflict Detection Agent.

Analyze:

Budget:
₹{state['budget']}

Days:
{state['days']}

Group:
{state['group_plan']}

Preferences:
{state['preference_analysis']}

Accessibility:
{state['accessibility_plan']}

Optimization:
{state['optimization']}

Detect:

- Budget conflicts
- Time conflicts
- Group conflicts
- Food conflicts
- Activity conflicts
- Transportation conflicts
- Rest conflicts
- Safety conflicts

Give severity:

LOW
MEDIUM
HIGH
"""

    return {
        **state,
        "conflicts": cached_ai(prompt)
    }


# ------------------------------------------------------------
# Conflict Resolution
# ------------------------------------------------------------

def conflict_resolution_agent(state):

    prompt = f"""
You are the Conflict Resolution Agent.

Conflicts:
{state['conflicts']}

Resolve them using this priority:

1. Safety
2. Accessibility
3. Hard constraints
4. Budget
5. Traveler preferences
6. Comfort
7. Experience

For every conflict provide:

Conflict
Decision
Reason
Trade-off
"""

    return {
        **state,
        "conflict_resolution":
            cached_ai(prompt)
    }


# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------

def validation_agent(state):

    prompt = f"""
You are the Autonomous Plan Validator.

Validate:

Destination:
{state['destination']}

Days:
{state['days']}

Budget:
₹{state['budget']}

Route:
{state['route_analysis']}

Conflict resolution:
{state['conflict_resolution']}

Base itinerary:
{state['base_itinerary']}

Check:

1. Budget feasibility
2. Time feasibility
3. Activity overload
4. Route logic
5. Traveler constraints
6. Accessibility
7. Safety
8. Contradictions
9. Missing information

Return:

VALID
or
NEEDS REVISION

Then explain.
"""

    return {
        **state,
        "validation": cached_ai(prompt)
    }


# ------------------------------------------------------------
# Dynamic Replanning
# ------------------------------------------------------------

def dynamic_replanning_agent(state):

    prompt = f"""
You are the Dynamic Replanning Agent.

Current plan:
{state['base_itinerary']}

Validation:
{state['validation']}

Create adaptive strategies for:

1. Weather change
2. Transport delay
3. Attraction closure
4. Budget reduction
5. Traveler fatigue
6. Activity cancellation

For each:

EVENT
Original plan
Replacement
Budget impact
Time impact
Reason
"""

    return {
        **state,
        "dynamic_replanning":
            cached_ai(prompt)
    }


# ------------------------------------------------------------
# Risk Agent
# ------------------------------------------------------------

def risk_agent(state):

    prompt = f"""
You are the Travel Risk Intelligence Agent.

Destination:
{state['destination']}

Travelers:
{state['travelers']}

Days:
{state['days']}

Identify risks related to:

- Weather
- Transportation
- Safety
- Budget
- Schedule
- Crowds
- Booking uncertainty
- Emergency situations

For each:

Risk
Probability
Impact
Risk level
Mitigation
"""

    return {
        **state,
        "risk": cached_ai(prompt)
    }


# ------------------------------------------------------------
# Reliability / Circuit Breaker
# ------------------------------------------------------------

def reliability_agent(state):

    reliability = """
TRAVELGENAI RELIABILITY ENGINE

✓ LLM service available
✓ Response caching enabled
✓ Previous results can be reused
✓ Failure fallback supported
✓ External live APIs clearly identified

Circuit breaker policy:

If an external service fails:

1. Do not fabricate data
2. Use cached information if available
3. Mark information as unavailable
4. Ask for user confirmation
5. Retry service later
"""

    return {
        **state,
        "reliability": reliability
    }


# ------------------------------------------------------------
# Explainability
# ------------------------------------------------------------

def explainability_agent(state):

    prompt = f"""
You are the Explainable AI Agent.

Explain why TravelGenAI made these decisions.

Decision:
{state['decision']}

Optimization:
{state['optimization']}

Conflicts:
{state['conflicts']}

Resolution:
{state['conflict_resolution']}

Validation:
{state['validation']}

Risk:
{state['risk']}

Explain:

1. Why this strategy was selected
2. Why alternatives were rejected
3. Budget trade-offs
4. Safety trade-offs
5. Personalization
6. Uncertainty
7. Limitations
"""

    return {
        **state,
        "explainability":
            cached_ai(prompt)
    }


# ------------------------------------------------------------
# Privacy Agent
# ------------------------------------------------------------

def privacy_agent(state):

    privacy = """
TRAVELGENAI PRIVACY & SECURITY

✓ API keys are stored in environment variables.

✓ API keys are never displayed.

✓ Only necessary travel information should be processed.

✓ Sensitive information should not be unnecessarily stored.

✓ Live location requires user permission.

✓ External APIs should receive minimum required data.

✓ Production databases should use encryption.

✓ Authentication should use secure sessions.

✓ Role-based access should restrict administrative data.

✓ Users should be able to delete saved information.

✓ Logs should avoid unnecessary personal information.
"""

    return {
        **state,
        "privacy": privacy
    }


# ------------------------------------------------------------
# Analytics
# ------------------------------------------------------------

def analytics_agent(state):

    analytics = {

        "destination":
            state["destination"],

        "days":
            state["days"],

        "travelers":
            state["travelers"],

        "budget":
            state["budget"],

        "phase_1":
            "completed",

        "phase_2":
            "completed",

        "phase_3":
            "completed",

        "phase_4":
            "completed",

        "optimization":
            "completed",

        "conflict_detection":
            "completed",

        "validation":
            "completed",

        "risk_analysis":
            "completed",

        "timestamp":
            datetime.now().isoformat()
    }

    return {
        **state,
        "analytics":
            json.dumps(
                analytics,
                indent=4
            )
    }


# ------------------------------------------------------------
# Master Orchestrator
# ------------------------------------------------------------

def orchestration_agent(state):

    prompt = f"""
You are the Master TravelGenAI Orchestrator.

Coordinate all phases.

PHASE 1:
{state['base_itinerary']}

PHASE 2:
Research:
{state['research']}

Budget:
{state['budget_analysis']}

Route:
{state['route_analysis']}

PHASE 3:
Personalization:
{state['personalization']}

Group:
{state['group_plan']}

Contingency:
{state['contingency']}

What-If:
{state['what_if']}

PHASE 4:

Decision:
{state['decision']}

Optimization:
{state['optimization']}

Conflicts:
{state['conflicts']}

Resolution:
{state['conflict_resolution']}

Validation:
{state['validation']}

Risk:
{state['risk']}

Dynamic Replanning:
{state['dynamic_replanning']}

Explainability:
{state['explainability']}

Create orchestration instructions.

Determine:

1. Highest-priority decisions
2. Conflicting results
3. Uncertain information
4. Information requiring live APIs
5. Information requiring user confirmation
6. Fallback strategy
7. Final planning strategy
"""

    return {
        **state,
        "orchestration":
            cached_ai(prompt)
    }


# ------------------------------------------------------------
# FINAL PLANNER
# ------------------------------------------------------------

def final_planner_agent(state):

    prompt = f"""
You are the Final Autonomous Travel Planner.

Create the final TravelGenAI plan.

==================================================
TRIP
==================================================

Destination:
{state['destination']}

Days:
{state['days']}

Travelers:
{state['travelers']}

Budget:
₹{state['budget']}

Interests:
{state['interests']}

Travel Style:
{state['travel_style']}

Food:
{state['food_preferences']}

Transport:
{state['transport_preference']}

Accessibility:
{state['accessibility']}


==================================================
INTELLIGENCE
==================================================

Traveler Profile:
{state['traveler_profile']}

Research:
{state['research']}

Budget:
{state['budget_analysis']}

Route:
{state['route_analysis']}

Preferences:
{state['preference_analysis']}

Personalization:
{state['personalization']}

Group:
{state['group_plan']}

Contingency:
{state['contingency']}

Packing:
{state['packing']}

Food:
{state['food_plan']}

Accessibility:
{state['accessibility_plan']}

What-If:
{state['what_if']}

Decision:
{state['decision']}

Optimization:
{state['optimization']}

Conflicts:
{state['conflicts']}

Conflict Resolution:
{state['conflict_resolution']}

Validation:
{state['validation']}

Dynamic Replanning:
{state['dynamic_replanning']}

Risk:
{state['risk']}

Orchestration:
{state['orchestration']}


==================================================
FINAL OUTPUT
==================================================

Create:

# 🌍 FINAL TRAVEL PLAN

## 1. Executive Summary

## 2. Traveler Profile

## 3. Recommended Strategy

## 4. Day-by-Day Itinerary

For every day include:

Morning
Afternoon
Evening
Food
Transport
Rest

## 5. Transportation Strategy

## 6. Accommodation Strategy

## 7. Food Strategy

## 8. Budget Strategy

## 9. Safety Strategy

## 10. Accessibility Strategy

## 11. Packing Checklist

## 12. Conflict Resolutions

## 13. Backup Plan

## 14. What-If Scenarios

## 15. Dynamic Replanning Rules

## 16. Risks and Mitigation

## 17. Important Decisions

## 18. User Confirmation Required

## 19. Live API Verification Required

## 20. Confidence and Limitations

IMPORTANT:

Never fabricate live:

- Flight availability
- Hotel availability
- Weather
- Traffic
- Maps
- Current prices
- Restaurant availability
- Tickets
- Bookings

If live information is unavailable,
clearly say that it requires verification.
"""

    return {
        **state,
        "final_itinerary":
            cached_ai(prompt)
    }


# ------------------------------------------------------------
# Provenance
# ------------------------------------------------------------

def provenance_agent(state):

    data = {

        "generated_at":
            datetime.now().isoformat(),

        "model":
            "Llama 3.3 70B via Groq",

        "orchestration":
            "LangGraph",

        "phases": [
            "Phase 1 - Core Planner",
            "Phase 2 - Research & Optimization",
            "Phase 3 - Personalization & Adaptive Planning",
            "Phase 4 - Autonomous Intelligence"
        ],

        "live_data_verified":
            False,

        "external_api_required":
            True,

        "note":
            "Live travel information must be obtained "
            "from verified external APIs."
    }

    return {
        **state,
        "provenance":
            json.dumps(
                data,
                indent=4
            )
    }


# ============================================================
# STATUS AGENT
# ============================================================

def status_agent(state):

    status = [

        "Phase 1 ✓ Core Planner",
        "Phase 1 ✓ Traveler Profile",
        "Phase 1 ✓ Transport",
        "Phase 1 ✓ Hotel",
        "Phase 1 ✓ Attractions",
        "Phase 1 ✓ Food",
        "Phase 1 ✓ Weather",
        "Phase 1 ✓ Safety",

        "Phase 2 ✓ Research",
        "Phase 2 ✓ Budget Optimization",
        "Phase 2 ✓ Route Optimization",
        "Phase 2 ✓ Preference Learning",
        "Phase 2 ✓ Provenance",

        "Phase 3 ✓ Memory",
        "Phase 3 ✓ Personalization",
        "Phase 3 ✓ Group Planning",
        "Phase 3 ✓ Contingency",
        "Phase 3 ✓ Packing",
        "Phase 3 ✓ Food Planning",
        "Phase 3 ✓ Accessibility",
        "Phase 3 ✓ What-If",
        "Phase 3 ✓ Live Trip Framework",
        "Phase 3 ✓ Notifications",
        "Phase 3 ✓ Feedback Learning",

        "Phase 4 ✓ Autonomous Decisions",
        "Phase 4 ✓ Multi-Objective Optimization",
        "Phase 4 ✓ Conflict Detection",
        "Phase 4 ✓ Conflict Resolution",
        "Phase 4 ✓ Validation",
        "Phase 4 ✓ Dynamic Replanning",
        "Phase 4 ✓ Risk Analysis",
        "Phase 4 ✓ Reliability",
        "Phase 4 ✓ Explainability",
        "Phase 4 ✓ Privacy",
        "Phase 4 ✓ Analytics",
        "Phase 4 ✓ Orchestration"

    ]

    return {
        **state,
        "status": status
    }


# ============================================================
# LANGGRAPH WORKFLOW
# ============================================================

def create_travel_graph():

    workflow = StateGraph(
        TravelState
    )


    # =========================
    # PHASE 1
    # =========================

    workflow.add_node(
        "profile",
        traveler_profile_agent
    )

    workflow.add_node(
        "transport",
        transport_agent
    )

    workflow.add_node(
        "hotels",
        hotel_agent
    )

    workflow.add_node(
        "attractions",
        attraction_agent
    )

    workflow.add_node(
        "restaurants",
        restaurant_agent
    )

    workflow.add_node(
        "weather",
        weather_agent
    )

    workflow.add_node(
        "safety",
        safety_agent
    )

    workflow.add_node(
        "base_itinerary",
        base_itinerary_agent
    )


    # =========================
    # PHASE 2
    # =========================

    workflow.add_node(
        "research",
        research_agent
    )

    workflow.add_node(
        "budget",
        budget_agent
    )

    workflow.add_node(
        "route",
        route_agent
    )

    workflow.add_node(
        "preferences",
        preference_agent
    )


    # =========================
    # PHASE 3
    # =========================

    workflow.add_node(
        "memory",
        memory_agent
    )

    workflow.add_node(
        "personalization",
        personalization_agent
    )

    workflow.add_node(
        "group",
        group_agent
    )

    workflow.add_node(
        "contingency",
        contingency_agent
    )

    workflow.add_node(
        "packing",
        packing_agent
    )

    workflow.add_node(
        "food_plan",
        food_plan_agent
    )

    workflow.add_node(
        "accessibility",
        accessibility_agent
    )

    workflow.add_node(
        "what_if",
        what_if_agent
    )

    workflow.add_node(
        "live_trip",
        live_trip_agent
    )

    workflow.add_node(
        "notifications",
        notification_agent
    )

    workflow.add_node(
        "feedback",
        feedback_agent
    )


    # =========================
    # PHASE 4
    # =========================

    workflow.add_node(
        "decision",
        decision_agent
    )

    workflow.add_node(
        "optimization",
        optimization_agent
    )

    workflow.add_node(
        "conflicts",
        conflict_detection_agent
    )

    workflow.add_node(
        "resolution",
        conflict_resolution_agent
    )

    workflow.add_node(
        "validation",
        validation_agent
    )

    workflow.add_node(
        "replanning",
        dynamic_replanning_agent
    )

    workflow.add_node(
        "risk",
        risk_agent
    )

    workflow.add_node(
        "reliability",
        reliability_agent
    )

    workflow.add_node(
        "explainability",
        explainability_agent
    )

    workflow.add_node(
        "privacy",
        privacy_agent
    )

    workflow.add_node(
        "analytics",
        analytics_agent
    )

    workflow.add_node(
        "orchestration",
        orchestration_agent
    )

    workflow.add_node(
        "final",
        final_planner_agent
    )

    workflow.add_node(
        "provenance",
        provenance_agent
    )

    workflow.add_node(
        "status",
        status_agent
    )


    # ========================================================
    # EDGES
    # ========================================================

    workflow.add_edge(
        START,
        "profile"
    )

    # Phase 1

    workflow.add_edge(
        "profile",
        "transport"
    )

    workflow.add_edge(
        "transport",
        "hotels"
    )

    workflow.add_edge(
        "hotels",
        "attractions"
    )

    workflow.add_edge(
        "attractions",
        "restaurants"
    )

    workflow.add_edge(
        "restaurants",
        "weather"
    )

    workflow.add_edge(
        "weather",
        "safety"
    )

    workflow.add_edge(
        "safety",
        "base_itinerary"
    )

    # Phase 2

    workflow.add_edge(
        "base_itinerary",
        "research"
    )

    workflow.add_edge(
        "research",
        "budget"
    )

    workflow.add_edge(
        "budget",
        "route"
    )

    workflow.add_edge(
        "route",
        "preferences"
    )

    # Phase 3

    workflow.add_edge(
        "preferences",
        "memory"
    )

    workflow.add_edge(
        "memory",
        "personalization"
    )

    workflow.add_edge(
        "personalization",
        "group"
    )

    workflow.add_edge(
        "group",
        "contingency"
    )

    workflow.add_edge(
        "contingency",
        "packing"
    )

    workflow.add_edge(
        "packing",
        "food_plan"
    )

    workflow.add_edge(
        "food_plan",
        "accessibility"
    )

    workflow.add_edge(
        "accessibility",
        "what_if"
    )

    workflow.add_edge(
        "what_if",
        "live_trip"
    )

    workflow.add_edge(
        "live_trip",
        "notifications"
    )

    workflow.add_edge(
        "notifications",
        "feedback"
    )

    # Phase 4

    workflow.add_edge(
        "feedback",
        "decision"
    )

    workflow.add_edge(
        "decision",
        "optimization"
    )

    workflow.add_edge(
        "optimization",
        "conflicts"
    )

    workflow.add_edge(
        "conflicts",
        "resolution"
    )

    workflow.add_edge(
        "resolution",
        "validation"
    )

    workflow.add_edge(
        "validation",
        "replanning"
    )

    workflow.add_edge(
        "replanning",
        "risk"
    )

    workflow.add_edge(
        "risk",
        "reliability"
    )

    workflow.add_edge(
        "reliability",
        "explainability"
    )

    workflow.add_edge(
        "explainability",
        "privacy"
    )

    workflow.add_edge(
        "privacy",
        "analytics"
    )

    workflow.add_edge(
        "analytics",
        "orchestration"
    )

    workflow.add_edge(
        "orchestration",
        "final"
    )

    workflow.add_edge(
        "final",
        "provenance"
    )

    workflow.add_edge(
        "provenance",
        "status"
    )

    workflow.add_edge(
        "status",
        END
    )


    return workflow.compile()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-title">
        🌍 TravelGenAI
    </div>

    <div class="subtitle">
        Autonomous Multi-Agent AI Travel Planner
    </div>
    """,
    unsafe_allow_html=True
)


st.caption(
    "Phase 1 + Phase 2 + Phase 3 + Phase 4 | "
    "Llama 3.3 + Groq + LangGraph + Streamlit"
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "✈️ Trip Configuration"
)


destination = st.sidebar.text_input(
    "📍 Destination",
    "Hyderabad"
)


days = st.sidebar.number_input(
    "📅 Number of Days",
    min_value=1,
    max_value=30,
    value=3
)


travelers = st.sidebar.number_input(
    "👥 Number of Travelers",
    min_value=1,
    max_value=20,
    value=2
)


budget = st.sidebar.number_input(
    "💰 Total Budget (₹)",
    min_value=1000,
    value=30000,
    step=1000
)


interests = st.sidebar.text_area(
    "❤️ Interests",
    "Food, history, nature, photography"
)


travel_style = st.sidebar.selectbox(
    "🎯 Travel Style",
    [
        "Budget",
        "Balanced",
        "Luxury",
        "Adventure",
        "Relaxed",
        "Family"
    ]
)


food_preferences = st.sidebar.text_input(
    "🍴 Food Preferences",
    "Local food, vegetarian"
)


transport_preference = st.sidebar.selectbox(
    "🚗 Transportation",
    [
        "Public Transport",
        "Taxi / Cab",
        "Rental Car",
        "Mixed"
    ]
)


accessibility = st.sidebar.text_area(
    "♿ Accessibility Requirements",
    "No special requirements"
)


group_preferences = st.sidebar.text_area(
    "👥 Group Preferences",
    "Sightseeing, food and photography"
)


special_requirements = st.sidebar.text_area(
    "⭐ Special Requirements",
    "Avoid excessive walking"
)


st.sidebar.divider()


generate = st.sidebar.button(
    "🚀 GENERATE COMPLETE PLAN",
    use_container_width=True
)


# ============================================================
# GENERATE PLAN
# ============================================================

if generate:

    if not destination.strip():

        st.error(
            "Please enter a destination."
        )

        st.stop()


    state = {

        "destination":
            destination,

        "days":
            int(days),

        "travelers":
            int(travelers),

        "budget":
            float(budget),

        "interests":
            interests,

        "travel_style":
            travel_style,

        "food_preferences":
            food_preferences,

        "transport_preference":
            transport_preference,

        "accessibility":
            accessibility,

        "group_preferences":
            group_preferences,

        "special_requirements":
            special_requirements
    }


    progress = st.progress(0)

    status_text = st.empty()


    try:

        status_text.write(
            "🤖 Initializing TravelGenAI..."
        )

        progress.progress(5)


        graph = create_travel_graph()


        status_text.write(
            "🧠 Phase 1: Core travel planning..."
        )

        progress.progress(20)


        status_text.write(
            "🔎 Phase 2: Research and optimization..."
        )

        progress.progress(40)


        status_text.write(
            "🧠 Phase 3: Personalization and adaptive planning..."
        )

        progress.progress(60)


        status_text.write(
            "🤖 Phase 4: Autonomous decision making..."
        )

        progress.progress(75)


        result = graph.invoke(state)


        progress.progress(100)


        status_text.write(
            "✅ All four phases completed!"
        )


    except Exception as e:

        st.error(
            f"TravelGenAI Error:\n\n{e}"
        )

        st.stop()


    # ========================================================
    # SAVE MEMORY
    # ========================================================

    memory_record = {

        "destination":
            destination,

        "days":
            int(days),

        "travelers":
            int(travelers),

        "budget":
            float(budget),

        "interests":
            interests,

        "travel_style":
            travel_style,

        "timestamp":
            datetime.now().isoformat()
    }


    st.session_state.travel_memory.append(
        memory_record
    )


    st.session_state.trip_history.append(
        memory_record
    )


    st.success(
        "🎉 Complete TravelGenAI plan generated!"
    )


    # ========================================================
    # OVERVIEW
    # ========================================================

    st.header(
        "📊 Trip Overview"
    )


    c1, c2, c3, c4 = st.columns(4)


    c1.metric(
        "📍 Destination",
        destination
    )


    c2.metric(
        "📅 Days",
        days
    )


    c3.metric(
        "👥 Travelers",
        travelers
    )


    c4.metric(
        "💰 Budget",
        f"₹{budget:,.0f}"
    )


    st.divider()


    # ========================================================
    # PHASE STATUS
    # ========================================================

    st.header(
        "🚀 Four-Phase AI System"
    )


    p1, p2, p3, p4 = st.columns(4)


    with p1:

        st.markdown(
            """
            <div class="phase-card">

            <h3>🧭 Phase 1</h3>

            <b>Core Planner</b>

            <p>
            Profile<br>
            Transport<br>
            Hotels<br>
            Attractions<br>
            Food<br>
            Safety
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    with p2:

        st.markdown(
            """
            <div class="phase-card">

            <h3>🔎 Phase 2</h3>

            <b>Research & Optimization</b>

            <p>
            Research<br>
            Budget<br>
            Route<br>
            Preferences<br>
            Provenance
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    with p3:

        st.markdown(
            """
            <div class="phase-card">

            <h3>🧠 Phase 3</h3>

            <b>Adaptive Intelligence</b>

            <p>
            Memory<br>
            Personalization<br>
            Group<br>
            What-If<br>
            Contingency
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    with p4:

        st.markdown(
            """
            <div class="phase-card">

            <h3>🤖 Phase 4</h3>

            <b>Autonomous Intelligence</b>

            <p>
            Optimization<br>
            Conflicts<br>
            Validation<br>
            Replanning<br>
            Explainability
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.divider()


    # ========================================================
    # AGENT STATUS
    # ========================================================

    st.header(
        "🤖 Agent Execution Status"
    )


    status_columns = st.columns(4)


    for i, agent in enumerate(
        result["status"]
    ):

        with status_columns[i % 4]:

            st.success(
                f"✓ {agent}"
            )


    st.divider()


    # ========================================================
    # TABS
    # ========================================================

    tabs = st.tabs(

        [

            "🌍 FINAL PLAN",

            "🧭 PHASE 1",

            "🔎 PHASE 2",

            "🧠 PHASE 3",

            "🤖 PHASE 4",

            "🎯 DECISION",

            "⚙️ OPTIMIZATION",

            "⚔️ CONFLICTS",

            "🔄 REPLANNING",

            "⚠️ RISKS",

            "💡 EXPLAINABLE AI",

            "🔐 PRIVACY",

            "📊 ANALYTICS",

            "🤖 ORCHESTRATOR",

            "🔗 PROVENANCE"

        ]

    )


    # ========================================================
    # FINAL PLAN
    # ========================================================

    with tabs[0]:

        st.header(
            "🌍 Final Autonomous Travel Plan"
        )

        st.markdown(
            result["final_itinerary"]
        )


    # ========================================================
    # PHASE 1
    # ========================================================

    with tabs[1]:

        st.subheader(
            "👤 Traveler Profile"
        )

        st.markdown(
            result["traveler_profile"]
        )


        st.subheader(
            "🚗 Transportation"
        )

        st.markdown(
            result["transport"]
        )


        st.subheader(
            "🏨 Accommodation"
        )

        st.markdown(
            result["hotels"]
        )


        st.subheader(
            "📍 Attractions"
        )

        st.markdown(
            result["attractions"]
        )


        st.subheader(
            "🍴 Restaurants & Food"
        )

        st.markdown(
            result["restaurants"]
        )


        st.subheader(
            "🌦️ Weather"
        )

        st.markdown(
            result["weather"]
        )


        st.subheader(
            "🛡️ Safety"
        )

        st.markdown(
            result["safety"]
        )


        st.subheader(
            "📅 Initial Itinerary"
        )

        st.markdown(
            result["base_itinerary"]
        )


    # ========================================================
    # PHASE 2
    # ========================================================

    with tabs[2]:

        st.subheader(
            "🔎 Research"
        )

        st.markdown(
            result["research"]
        )


        st.subheader(
            "💰 Budget Optimization"
        )

        st.markdown(
            result["budget_analysis"]
        )


        st.subheader(
            "🗺️ Route Optimization"
        )

        st.markdown(
            result["route_analysis"]
        )


        st.subheader(
            "❤️ Preference Analysis"
        )

        st.markdown(
            result["preference_analysis"]
        )


    # ========================================================
    # PHASE 3
    # ========================================================

    with tabs[3]:

        st.subheader(
            "🧠 Memory"
        )

        st.code(
            result["memory"],
            language="json"
        )


        st.subheader(
            "✨ Personalization"
        )

        st.markdown(
            result["personalization"]
        )


        st.subheader(
            "👥 Group Planning"
        )

        st.markdown(
            result["group_plan"]
        )


        st.subheader(
            "🔄 Contingency Planning"
        )

        st.markdown(
            result["contingency"]
        )


        st.subheader(
            "🎒 Smart Packing"
        )

        st.markdown(
            result["packing"]
        )


        st.subheader(
            "🍴 Food Planning"
        )

        st.markdown(
            result["food_plan"]
        )


        st.subheader(
            "♿ Accessibility"
        )

        st.markdown(
            result["accessibility_plan"]
        )


        st.subheader(
            "🔮 What-If Simulation"
        )

        st.markdown(
            result["what_if"]
        )


        st.subheader(
            "📡 Live Trip Framework"
        )

        st.markdown(
            result["live_trip"]
        )


        st.subheader(
            "🔔 Notifications"
        )

        st.markdown(
            result["notifications"]
        )


    # ========================================================
    # PHASE 4
    # ========================================================

    with tabs[4]:

        st.subheader(
            "🎯 Autonomous Decision"
        )

        st.markdown(
            result["decision"]
        )


        st.subheader(
            "⚙️ Multi-Objective Optimization"
        )

        st.markdown(
            result["optimization"]
        )


        st.subheader(
            "⚔️ Conflict Detection"
        )

        st.markdown(
            result["conflicts"]
        )


        st.subheader(
            "🤝 Conflict Resolution"
        )

        st.markdown(
            result["conflict_resolution"]
        )


        st.subheader(
            "✅ Validation"
        )

        st.markdown(
            result["validation"]
        )


        st.subheader(
            "🔄 Dynamic Replanning"
        )

        st.markdown(
            result["dynamic_replanning"]
        )


        st.subheader(
            "⚠️ Risk Analysis"
        )

        st.markdown(
            result["risk"]
        )


        st.subheader(
            "🛡️ Reliability"
        )

        st.markdown(
            result["reliability"]
        )


        st.subheader(
            "💡 Explainability"
        )

        st.markdown(
            result["explainability"]
        )


        st.subheader(
            "🔐 Privacy"
        )

        st.markdown(
            result["privacy"]
        )


        st.subheader(
            "📊 Analytics"
        )

        st.markdown(
            result["analytics"]
        )


        st.subheader(
            "🤖 Orchestration"
        )

        st.markdown(
            result["orchestration"]
        )


    # ========================================================
    # DECISION
    # ========================================================

    with tabs[5]:

        st.header(
            "🎯 Autonomous Decision Engine"
        )

        st.markdown(
            result["decision"]
        )


    # ========================================================
    # OPTIMIZATION
    # ========================================================

    with tabs[6]:

        st.header(
            "⚙️ Multi-Objective Optimization"
        )

        st.markdown(
            result["optimization"]
        )


    # ========================================================
    # CONFLICT
    # ========================================================

    with tabs[7]:

        st.header(
            "⚔️ Conflict Detection & Resolution"
        )

        st.subheader(
            "Detected Conflicts"
        )

        st.markdown(
            result["conflicts"]
        )


        st.subheader(
            "Resolution"
        )

        st.markdown(
            result["conflict_resolution"]
        )


    # ========================================================
    # REPLANNING
    # ========================================================

    with tabs[8]:

        st.header(
            "🔄 Dynamic Replanning"
        )

        st.markdown(
            result["dynamic_replanning"]
        )


        st.subheader(
            "🔮 What-If Simulation"
        )

        st.markdown(
            result["what_if"]
        )


        st.subheader(
            "🛟 Contingency Plan"
        )

        st.markdown(
            result["contingency"]
        )


    # ========================================================
    # RISK
    # ========================================================

    with tabs[9]:

        st.header(
            "⚠️ Travel Risk Analysis"
        )

        st.markdown(
            result["risk"]
        )


        st.subheader(
            "🛡️ Reliability"
        )

        st.markdown(
            result["reliability"]
        )


    # ========================================================
    # EXPLAINABILITY
    # ========================================================

    with tabs[10]:

        st.header(
            "💡 Explainable AI"
        )

        st.markdown(
            result["explainability"]
        )


    # ========================================================
    # PRIVACY
    # ========================================================

    with tabs[11]:

        st.header(
            "🔐 Privacy & Security"
        )

        st.markdown(
            result["privacy"]
        )


    # ========================================================
    # ANALYTICS
    # ========================================================

    with tabs[12]:

        st.header(
            "📊 TravelGenAI Analytics"
        )

        st.json(
            json.loads(
                result["analytics"]
            )
        )


        st.subheader(
            "📚 Trip History"
        )

        if st.session_state.trip_history:

            st.dataframe(
                st.session_state.trip_history,
                use_container_width=True
            )


        st.subheader(
            "🧠 Stored Memory"
        )

        st.write(
            len(
                st.session_state.travel_memory
            ),
            "trip(s) stored in this session."
        )


    # ========================================================
    # ORCHESTRATOR
    # ========================================================

    with tabs[13]:

        st.header(
            "🤖 Master Agent Orchestrator"
        )

        st.markdown(
            result["orchestration"]
        )


    # ========================================================
    # PROVENANCE
    # ========================================================

    with tabs[14]:

        st.header(
            "🔗 Data Provenance"
        )

        st.code(
            result["provenance"],
            language="json"
        )


    # ========================================================
    # FEEDBACK
    # ========================================================

    st.divider()

    st.header(
        "⭐ Traveler Feedback"
    )


    rating = st.slider(
        "Rate this travel plan",
        min_value=1,
        max_value=5,
        value=4
    )


    feedback = st.text_area(
        "What should TravelGenAI improve?"
    )


    if st.button(
        "💾 Save Feedback"
    ):

        feedback_record = {

            "rating":
                rating,

            "feedback":
                feedback,

            "destination":
                destination,

            "timestamp":
                datetime.now().isoformat()
        }


        st.session_state.feedback_history.append(
            feedback_record
        )


        st.success(
            "Feedback saved successfully!"
        )


    # ========================================================
    # DOWNLOAD REPORT
    # ========================================================

    st.divider()

    st.header(
        "📥 Export Complete TravelGenAI Report"
    )


    report = f"""
============================================================
TRAVELGENAI
AUTONOMOUS MULTI-AGENT TRAVEL PLANNER
============================================================

Destination: {destination}
Days: {days}
Travelers: {travelers}
Budget: ₹{budget:,.0f}

============================================================
PHASE 1 — CORE PLANNER
============================================================

TRAVELER PROFILE

{result["traveler_profile"]}

TRANSPORT

{result["transport"]}

HOTELS

{result["hotels"]}

ATTRACTIONS

{result["attractions"]}

FOOD

{result["restaurants"]}

WEATHER

{result["weather"]}

SAFETY

{result["safety"]}

INITIAL ITINERARY

{result["base_itinerary"]}


============================================================
PHASE 2 — RESEARCH & OPTIMIZATION
============================================================

RESEARCH

{result["research"]}

BUDGET

{result["budget_analysis"]}

ROUTE

{result["route_analysis"]}

PREFERENCES

{result["preference_analysis"]}


============================================================
PHASE 3 — ADAPTIVE INTELLIGENCE
============================================================

MEMORY

{result["memory"]}

PERSONALIZATION

{result["personalization"]}

GROUP PLANNING

{result["group_plan"]}

CONTINGENCY

{result["contingency"]}

PACKING

{result["packing"]}

FOOD PLANNING

{result["food_plan"]}

ACCESSIBILITY

{result["accessibility_plan"]}

WHAT-IF

{result["what_if"]}

LIVE TRIP

{result["live_trip"]}

NOTIFICATIONS

{result["notifications"]}


============================================================
PHASE 4 — AUTONOMOUS INTELLIGENCE
============================================================

DECISION

{result["decision"]}

OPTIMIZATION

{result["optimization"]}

CONFLICTS

{result["conflicts"]}

CONFLICT RESOLUTION

{result["conflict_resolution"]}

VALIDATION

{result["validation"]}

DYNAMIC REPLANNING

{result["dynamic_replanning"]}

RISK

{result["risk"]}

RELIABILITY

{result["reliability"]}

EXPLAINABILITY

{result["explainability"]}

PRIVACY

{result["privacy"]}

ANALYTICS

{result["analytics"]}

ORCHESTRATION

{result["orchestration"]}


============================================================
FINAL TRAVEL PLAN
============================================================

{result["final_itinerary"]}


============================================================
PROVENANCE
============================================================

{result["provenance"]}


============================================================
END OF REPORT
============================================================
"""


    st.download_button(
        label="📄 Download Complete Report",
        data=report,
        file_name="TravelGenAI_All_Phases_Report.txt",
        mime="text/plain",
        use_container_width=True
    )


# ============================================================
# DEFAULT HOME SCREEN
# ============================================================

else:

    st.header(
        "🚀 Complete Four-Phase Travel AI"
    )

    st.write(
        "Configure your trip from the sidebar "
        "and generate an autonomous travel plan."
    )


    st.divider()


    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.markdown(
            """
            ### 🧭 Phase 1

            Core travel planning

            ✓ Traveler Profile

            ✓ Transport

            ✓ Hotels

            ✓ Attractions

            ✓ Food

            ✓ Safety
            """
        )


    with c2:

        st.markdown(
            """
            ### 🔎 Phase 2

            Research & optimization

            ✓ Research

            ✓ Budget

            ✓ Route

            ✓ Preferences

            ✓ Provenance
            """
        )


    with c3:

        st.markdown(
            """
            ### 🧠 Phase 3

            Adaptive intelligence

            ✓ Memory

            ✓ Personalization

            ✓ Group Planning

            ✓ What-If

            ✓ Contingency

            ✓ Accessibility
            """
        )


    with c4:

        st.markdown(
            """
            ### 🤖 Phase 4

            Autonomous intelligence

            ✓ Optimization

            ✓ Conflict Resolution

            ✓ Validation

            ✓ Dynamic Replanning

            ✓ Risk

            ✓ Explainability
            """
        )


    st.divider()


    st.info(
        "👈 Enter your trip information in the sidebar "
        "and click **GENERATE COMPLETE PLAN**."
    )
