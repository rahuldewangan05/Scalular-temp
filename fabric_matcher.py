import google.generativeai as genai
import json
import os
from dotenv import load_dotenv
import re

load_dotenv()

class TechPackAnalyzer:
    def __init__(self):
        # Initialize Gemini
        api_key = os.environ['api_key']
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Load object.json
        with open("object.json", 'r') as f:
            self.fabric_data = json.load(f)
    
    def _get_available_categories(self):
        """Get list of available categories from object.json."""
        return list(self.fabric_data.keys())
    
    def _get_available_gms_ranges(self, category, fabric_blend=None):
        """Get available GSM ranges for a category and fabric blend."""
        if category not in self.fabric_data:
            return []
        
        if fabric_blend and fabric_blend in self.fabric_data[category]:
            # If fabric blend is provided and exists, return its GSM ranges
            return list(self.fabric_data[category][fabric_blend].keys())
        
        # Get all unique GSM ranges across all fabrics in the category
        gms_ranges = set()
        for fabric in self.fabric_data[category].values():
            gms_ranges.update(fabric.keys())
        return sorted(list(gms_ranges))
    
    def _get_available_fabrics(self, category):
        """Get available fabrics for a category."""
        if category not in self.fabric_data:
            return []
        return list(self.fabric_data[category].keys())
    
    def _normalize_size_range(self, sizes):
        """Normalize size range to format like 'S-3XL' or 'M-XXXL'."""
        if not sizes:
            return "M"  # Default size
            
        # Define size order for proper sorting
        size_order = {
            'XXS': 0,
            'XS': 1,
            'S': 2,
            'M': 3,
            'L': 4,
            'XL': 5,
            'XXL': 6,
            'XXXL': 7,
            'XXXXL': 8
        }
        
        # Convert all sizes to a standard format
        size_mapping = {
            '1X': 'XL',
            '2X': 'XXL',
            '3X': 'XXXL',
            '4X': 'XXXXL'
        }
        
        normalized_sizes = []
        for size in sizes:
            size = size.strip().upper()
            if size in size_mapping:
                normalized_sizes.append(size_mapping[size])
            else:
                normalized_sizes.append(size)
        
        # Remove duplicates and sort using custom order
        unique_sizes = list(set(normalized_sizes))
        unique_sizes.sort(key=lambda x: size_order.get(x, 999))  # 999 for unknown sizes
        
        if len(unique_sizes) == 1:
            return unique_sizes[0]
        
        # Get smallest and largest sizes
        smallest = unique_sizes[0]
        largest = unique_sizes[-1]
        
        return f"{smallest}-{largest}"
    
    def _create_extraction_prompt(self, text_data):
        """Create prompt for initial information extraction."""
        prompt = f"""You are a technical pack analysis expert. Analyze the following text and extract specific information:

Text to analyze:
{text_data}

Extract the following information in JSON format:
{{
    "gender": "men/women/kids (based on text)",
    "product_name": "extract product name if available",
    "zipper": true/false (if zipper information is found),
    "logo_embroidery": true/false (if logo or embroidery information is found),
    "sizes": ["list of all sizes found (e.g., ['S', 'M', 'L', 'XL', '2X', '3X'])"],
    "print": "extract print type if available then matches with any o from the list ['Waterprint', "Puff Print", "HD Print", "Foil", "Sublimation", "Tie Die","Multi"], otherwise 'solid'",
    "category": "extract category information",
    "gms": "extract GSM/GMS/weight information if available (look for terms like GSM, GMS, weight, etc.)"
}}

Important notes:
1. For sizes, list ALL sizes found in the text
2. For GSM/GMS, look for terms like 'GSM', 'GMS', 'weight', 'fabric weight', etc.
3. Return only the JSON object, nothing else."""

        return prompt
    
    def _create_category_matching_prompt(self, extracted_category, available_categories):
        """Create prompt for category matching."""
        prompt = f"""You are a category matching expert. Match the extracted category with the most appropriate category from the available options.

Extracted category: {extracted_category}

Available categories:
{json.dumps(available_categories, indent=2)}

Return your response in this JSON format:
{{
    "matched_category": "exact category name from available list",
    "confidence": "high/medium/low",
    "reasoning": "brief explanation of why this category was matched"
}}"""

        return prompt
    
    def _create_gms_matching_prompt(self, extracted_gms, available_ranges):
        """Create prompt for GSM range matching."""
        prompt = f"""You are a GSM matching expert. Match the extracted GSM value with the most appropriate range from the available options.

Extracted GSM/GMS: {extracted_gms}

Available GSM ranges:
{json.dumps(available_ranges, indent=2)}

Return your response in this JSON format:
{{
    "matched_gms_range": "exact GSM range from available list",
    "confidence": "high/medium/low",
    "reasoning": "brief explanation of why this range was matched"
}}"""

        return prompt
    
    def _create_fabric_matching_prompt(self, text_data, category, available_fabrics):
        """Create prompt for fabric matching."""
        prompt = f"""You are a fabric matching expert. Analyze the text and match it with the most appropriate fabric from the available options.

Text to analyze:
{text_data}

Category: {category}

Available fabrics:
{json.dumps(available_fabrics, indent=2)}

Consider the following when matching:
1. Look for fabric composition percentages (e.g., "60% cotton, 40% polyester")
2. Look for fabric types (e.g., "Single Jersey", "Lycra", "Pique")
3. Look for fabric weights or specifications

Return your response in this JSON format:
{{
    "matched_fabric": "exact fabric name from available list",
    "confidence": "high/medium/low",
    "reasoning": "brief explanation of why this fabric was matched"
}}"""

        return prompt
    
    def _extract_json_from_response(self, response_text):
        """Extract JSON from Gemini response."""
        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return None
        except json.JSONDecodeError:
            return None
    
    def _get_gms_from_object(self, category, fabric_blend):
        """Get GSM range directly from object.json for a given category and fabric blend."""
        try:
            if (category in self.fabric_data and 
                fabric_blend in self.fabric_data[category] and 
                self.fabric_data[category][fabric_blend]):
                # Get the first GSM range available for this fabric
                return list(self.fabric_data[category][fabric_blend].keys())[0]
            return None
        except Exception as e:
            print(f"Error getting GSM from object.json: {e}")
            return None
    
    async def analyze_techpack(self, text_data):
        """Main function to analyze techpack and extract all required information."""
        try:
            # Step 1: Extract basic information
            basic_info_prompt = self._create_extraction_prompt(text_data)
            basic_info_response = self.model.generate_content(basic_info_prompt)
            basic_info = self._extract_json_from_response(basic_info_response.text)
            
            if not basic_info:
                raise Exception("Failed to extract basic information")
            
            # Step 2: Match category
            available_categories = self._get_available_categories()
            category_prompt = self._create_category_matching_prompt(basic_info.get('category', ''), available_categories)
            category_response = self.model.generate_content(category_prompt)
            category_match = self._extract_json_from_response(category_response.text)
            
            if not category_match:
                raise Exception("Failed to match category")
            
            matched_category = category_match['matched_category']
            
            # Step 3: Match fabric
            available_fabrics = self._get_available_fabrics(matched_category)
            fabric_prompt = self._create_fabric_matching_prompt(text_data, matched_category, available_fabrics)
            fabric_response = self.model.generate_content(fabric_prompt)
            fabric_match = self._extract_json_from_response(fabric_response.text)
            
            if not fabric_match:
                raise Exception("Failed to match fabric")
            
            matched_fabric = fabric_match['matched_fabric']
            
            # Step 4: Get GSM range
            # First try to get GSM directly from object.json
            matched_gms = self._get_gms_from_object(matched_category, matched_fabric)
            
            # If not found in object.json, use default
            if not matched_gms:
                matched_gms = "160-180"
            
            # Normalize size range
            sizes = basic_info.get('sizes', [])
            normalized_size = self._normalize_size_range(sizes)
            
            # Combine all results
            result = {
                "gender": basic_info.get('gender', 'men'),
                "product_name": basic_info.get('product_name', ''),
                "zipper": basic_info.get('zipper', False),
                "logo_embroidery": basic_info.get('logo_embroidery', False),
                "size": normalized_size,
                "print": basic_info.get('print', 'solid'),
                "category": matched_category,
                "quantity_in_gms": matched_gms,
                "fabric_and_blend": matched_fabric
            }
            
            return result
            
        except Exception as e:
            print(f"Error in techpack analysis: {e}")
            # Return default values if analysis fails
            return {
                "gender": "men",
                "product_name": "",
                "zipper": False,
                "logo_embroidery": False,
                "size": "M",
                "print": "solid",
                "category": "crewneck-t-shirts",
                "quantity_in_gms": "160-180",
                "fabric_and_blend": "Single-Jersey-(Combed)"
            }

# Example usage
async def main():
    analyzer = TechPackAnalyzer()
    
    # Example text
    text = """
    Product: Men's Crewneck T-shirt
    Fabric: 100% Cotton Single Jersey
    Weight: 180 GSM
    Sizes: S, M, L, XL, 2X, 3X
    Features: Logo embroidery on chest
    Print: Solid color
    """
    
    result = await analyzer.analyze_techpack(text)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 