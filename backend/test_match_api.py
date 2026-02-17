"""Test the profile-based matching API endpoint.

This demonstrates how the frontend should call the /api/match endpoint
with a structured user profile to get deterministic match scores.
"""

import requests
import json


# API endpoint
API_URL = "http://localhost:5000/api/match"


# Example user profile from questionnaire
sample_profile = {
    "skills": ["Python", "Marketing", "Sales", "Data Analysis"],
    "experience_level": "Intermediate",
    "industry_interest": ["AI/ML", "E-commerce", "SaaS"],
    "business_model_preference": "B2B",
    "starting_capital": 5000,
    "desired_income": 10000,
    "time_commitment": "part_time",  # Options: full_time, part_time, flexible
    "network_strength": "Moderate",
    "existing_assets": ["website", "social media following"]
}


def test_match_endpoint():
    """Test the /api/match endpoint with a sample profile."""
    print("=" * 60)
    print("Testing Profile-Based Matching API")
    print("=" * 60)
    
    print("\n📋 User Profile:")
    print(json.dumps(sample_profile, indent=2))
    
    print("\n🔄 Calling API...")
    
    try:
        response = requests.post(
            API_URL,
            json=sample_profile,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"\n✅ Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"\n📊 Results:")
            print(f"  Total candidates: {data.get('total_candidates')}")
            print(f"  Filtered candidates: {data.get('filtered_candidates')}")
            print(f"  Matches returned: {len(data.get('matches', []))}")
            print(f"  Generation time: {data.get('generation_time_ms', 0):.2f}ms")
            
            # Show top 3 matches
            matches = data.get('matches', [])
            if matches:
                print("\n🎯 Top 3 Matches:")
                for i, match in enumerate(matches[:3], 1):
                    print(f"\n  #{i} - {match['match_percentage']}% Match")
                    print(f"     Domain: {match['domain']}")
                    print(f"     Business Model: {match['business_model']}")
                    print(f"     Difficulty: {match['difficulty']}")
                    print(f"     Scalability: {match['scalability']}")
                    print(f"     Cost: {match['estimated_cost_bucket']}")
                    print(f"     Summary: {match['short_summary'][:100]}...")
                    print(f"     Explanation: {match['explanation']}")
                    
                    # Show score breakdown
                    breakdown = match.get('score_breakdown', {})
                    if breakdown:
                        print(f"     Score Breakdown:")
                        for component, score in breakdown.items():
                            print(f"       - {component}: {score:.2f}")
            
        else:
            print(f"\n❌ Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("\n❌ Could not connect to API. Make sure the server is running:")
        print("   python app.py")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")


def test_validation_errors():
    """Test validation with invalid profile."""
    print("\n" + "=" * 60)
    print("Testing Validation Errors")
    print("=" * 60)
    
    invalid_profile = {
        "skills": ["Python"],
        "experience_level": "InvalidLevel",  # Should fail
        "starting_capital": -1000  # Should fail
    }
    
    print("\n📋 Invalid Profile:")
    print(json.dumps(invalid_profile, indent=2))
    
    print("\n🔄 Calling API...")
    
    try:
        response = requests.post(
            API_URL,
            json=invalid_profile,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"\n✅ Status: {response.status_code}")
        if response.status_code == 400:
            print("✓ Validation error detected as expected")
            error_data = response.json()
            print(f"\nError details:")
            print(json.dumps(error_data, indent=2))
        else:
            print("⚠️ Expected 400 validation error")
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    print("🚀 Starting API Tests\n")
    
    # Test main endpoint
    test_match_endpoint()
    
    # Test validation
    # test_validation_errors()
    
    print("\n" + "=" * 60)
    print("✅ Tests Complete")
    print("=" * 60)
