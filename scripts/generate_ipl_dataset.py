
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IPLDatasetGenerator:
    
    def __init__(self, data_dir: str):
       
        self.data_dir = Path(data_dir)
        self.batting_stats_dir = self.data_dir / "batting_stats"
        self.bowling_stats_dir = self.data_dir / "bowling_stats"
        self.match_info_dir = self.data_dir / "match_info"
        self.scorecards_dir = self.data_dir / "scorecards"
        self.squads_dir = self.data_dir / "squads"
        self.standings_dir = self.data_dir / "standings"
        self.team_stats_dir = self.data_dir / "team_stats"
        self.teams_dir = self.data_dir / "teams"
        self.matches_dir = self.data_dir / "matches"
        self.player_career_dir = self.data_dir / "player_career_stats"
        self.match_commentary_dir = self.data_dir / "match_innings_commentary"
        self.match_live_dir = self.data_dir / "match_live_details"
        self.match_wagon_dir = self.data_dir / "match_wagon_wheel"
        self.questions_answers = []
        
    def load_json_file(self, filepath: Path) -> Dict[str, Any]:
 
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load {filepath}: {e}")
            return {}
    
    def extract_data(self, data: Any, *keys) -> Any:
        
        if not isinstance(data, dict):
            return None
        
        # Try direct access first
        current = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                current = None
                break
        
        if current is not None:
            return current
        
        # Try with 'response' wrapper
        if 'response' in data:
            current = data['response']
            for key in keys:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    return None
            return current
        
        return None
    
    
    def load_json_file(self, filepath: Path) -> Dict[str, Any]:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load {filepath}: {e}")
            return {}
    
    def generate_batting_qa(self) -> List[Tuple[str, str]]:
        qa_pairs = []
        
        # Process ALL batting stats files
        batting_files = {
            "batting_most_runs.json": "most runs",
            "batting_highest_average.json": "highest average",
            "batting_most_run100.json": "most centuries",
            "batting_most_run50.json": "most fifties",
            "batting_highest_strikerate.json": "highest strike rate",
            "batting_most_run6.json": "most sixes",
            "batting_most_run4.json": "most fours",
            "batting_highest_strikerate_innings.json": "highest strike rate in an innings",
            "batting_most_runs_innings.json": "most runs in an innings",
            "batting_most_run6_innings.json": "most sixes in an innings",
            "batting_most_run4_innings.json": "most fours in an innings"
        }
        
        for filename, stat_type in batting_files.items():
            filepath = self.batting_stats_dir / filename
            if not filepath.exists():
                continue
                
            data = self.load_json_file(filepath)
            if not data.get('response', {}).get('stats'):
                continue
            
            stats = data['response']['stats'][:10]  # Top 10 for each category
            
            for i, stat in enumerate(stats):
                player_name = stat.get('player', {}).get('title', 'Unknown')
                team_name = stat.get('team', {}).get('title', 'Unknown')
                
                # Most Runs
                if "most_runs.json" in filename:
                    runs = stat.get('runs', 0)
                    matches = stat.get('matches', 0)
                    avg = stat.get('average', 0)
                    strike_rate = stat.get('strike', 0)
                    
                    if i == 0:
                        qa_pairs.append((
                            f"Who scored the most runs in IPL 2022?",
                            f"{player_name} from {team_name} scored {runs} runs in {matches} matches with an average of {avg} and strike rate of {strike_rate}."
                        ))
                    
                    qa_pairs.append((
                        f"How many runs did {player_name} score in IPL 2022?",
                        f"{player_name} scored {runs} runs in {matches} matches for {team_name} with a batting average of {avg}."
                    ))
                    
                    qa_pairs.append((
                        f"What is {player_name}'s batting average in IPL 2022?",
                        f"{player_name}'s batting average in IPL 2022 is {avg}. He scored {runs} runs in {matches} matches."
                    ))
                
                # Highest Average
                elif "highest_average.json" in filename:
                    avg = stat.get('average', 0)
                    runs = stat.get('runs', 0)
                    matches = stat.get('matches', 0)
                    innings = stat.get('innings', 0)
                    
                    if i == 0:
                        qa_pairs.append((
                            f"Which batsman has the highest batting average in IPL 2022?",
                            f"{player_name} from {team_name} has the highest average of {avg}, scoring {runs} runs in {innings} innings across {matches} matches."
                        ))
                    
                    qa_pairs.append((
                        f"What is the batting average of {player_name} in IPL 2022?",
                        f"{player_name} has an excellent average of {avg} in IPL 2022, scoring {runs} runs for {team_name}."
                    ))
                
                # Most Centuries
                elif "most_run100.json" in filename:
                    centuries = stat.get('run100', 0)
                    runs = stat.get('runs', 0)
                    matches = stat.get('matches', 0)
                    
                    if centuries > 0:
                        qa_pairs.append((
                            f"How many centuries did {player_name} score in IPL 2022?",
                            f"{player_name} from {team_name} scored {centuries} century(ies) in IPL 2022, accumulating {runs} total runs in {matches} matches."
                        ))
                        
                        if i == 0:
                            qa_pairs.append((
                                f"Who scored the most centuries in IPL 2022?",
                                f"{player_name} from {team_name} scored {centuries} century(ies), the most in IPL 2022."
                            ))
                
                # Most Fifties
                elif "most_run50.json" in filename:
                    fifties = stat.get('run50', 0)
                    runs = stat.get('runs', 0)
                    matches = stat.get('matches', 0)
                    
                    if fifties > 0:
                        qa_pairs.append((
                            f"How many fifties did {player_name} score in IPL 2022?",
                            f"{player_name} from {team_name} scored {fifties} fifty(ies) in {matches} matches with {runs} total runs."
                        ))
                        
                        if i == 0:
                            qa_pairs.append((
                                f"Who scored the most fifties in IPL 2022?",
                                f"{player_name} from {team_name} scored {fifties} fifties in IPL 2022."
                            ))
                
                # Highest Strike Rate
                elif "highest_strikerate.json" in filename and "innings" not in filename:
                    strike_rate = stat.get('strike', 0)
                    runs = stat.get('runs', 0)
                    balls = stat.get('balls', 0)
                    
                    if i == 0:
                        qa_pairs.append((
                            f"Which batsman has the highest strike rate in IPL 2022?",
                            f"{player_name} from {team_name} has the highest strike rate of {strike_rate}, scoring {runs} runs off {balls} balls."
                        ))
                    
                    qa_pairs.append((
                        f"What is {player_name}'s strike rate in IPL 2022?",
                        f"{player_name} has a strike rate of {strike_rate}, having scored {runs} runs from {balls} balls for {team_name}."
                    ))
                
                # Most Sixes
                elif "most_run6.json" in filename and "innings" not in filename:
                    sixes = stat.get('run6', 0)
                    runs = stat.get('runs', 0)
                    matches = stat.get('matches', 0)
                    
                    if i == 0:
                        qa_pairs.append((
                            f"Who hit the most sixes in IPL 2022?",
                            f"{player_name} from {team_name} hit the most sixes with {sixes} sixes while scoring {runs} runs."
                        ))
                    
                    qa_pairs.append((
                        f"How many sixes did {player_name} hit in IPL 2022?",
                        f"{player_name} hit {sixes} sixes in {matches} matches for {team_name}, scoring {runs} total runs."
                    ))
                
                # Most Fours
                elif "most_run4.json" in filename and "innings" not in filename:
                    fours = stat.get('run4', 0)
                    runs = stat.get('runs', 0)
                    matches = stat.get('matches', 0)
                    
                    if i == 0:
                        qa_pairs.append((
                            f"Who hit the most fours in IPL 2022?",
                            f"{player_name} from {team_name} hit the most fours with {fours} fours while scoring {runs} runs."
                        ))
                    
                    qa_pairs.append((
                        f"How many fours did {player_name} hit in IPL 2022?",
                        f"{player_name} hit {fours} fours in {matches} matches for {team_name}."
                    ))
                
                # Innings-specific records
                elif "innings" in filename:
                    runs = stat.get('runs', 0)
                    balls = stat.get('balls', 0)
                    strike = stat.get('strike', 0)
                    
                    if "most_runs_innings" in filename:
                        qa_pairs.append((
                            f"What is {player_name}'s highest score in an innings in IPL 2022?",
                            f"{player_name}'s highest score in an innings is {runs} runs off {balls} balls for {team_name}."
                        ))
                    elif "strikerate_innings" in filename:
                        qa_pairs.append((
                            f"What is {player_name}'s best strike rate in an innings in IPL 2022?",
                            f"{player_name} achieved a strike rate of {strike} in an innings, scoring {runs} runs off {balls} balls."
                        ))
                    elif "run6_innings" in filename:
                        sixes = stat.get('run6', 0)
                        qa_pairs.append((
                            f"What is the most sixes {player_name} hit in a single innings in IPL 2022?",
                            f"{player_name} hit {sixes} sixes in a single innings, scoring {runs} runs for {team_name}."
                        ))
                    elif "run4_innings" in filename:
                        fours = stat.get('run4', 0)
                        qa_pairs.append((
                            f"What is the most fours {player_name} hit in a single innings in IPL 2022?",
                            f"{player_name} hit {fours} fours in a single innings, scoring {runs} runs."
                        ))
        
        return qa_pairs
    
    
    def generate_bowling_qa(self) -> List[Tuple[str, str]]:
        """Generate Q&A pairs from ALL bowling statistics files."""
        qa_pairs = []
        
        # Process ALL bowling stats files
        bowling_files = {
            "bowling_top_wicket_takers.json": "top wicket takers",
            "bowling_best_economy_rates.json": "best economy rates",
            "bowling_best_averages.json": "best bowling averages",
            "bowling_best_bowling_figures.json": "best bowling figures",
            "bowling_best_strike_rates.json": "best strike rates",
            "bowling_five_wickets.json": "five-wicket hauls",
            "bowling_four_wickets.json": "four-wicket hauls",
            "bowling_maidens.json": "maiden overs",
            "bowling_most_runs_conceded_innings.json": "most runs conceded in innings",
            "bowling_best_economy_rates_innings.json": "best economy in innings",
            "bowling_best_strike_rates_innings.json": "best strike rate in innings"
        }
        
        for filename, stat_type in bowling_files.items():
            filepath = self.bowling_stats_dir / filename
            if not filepath.exists():
                continue
                
            data = self.load_json_file(filepath)
            if not data.get('response', {}).get('stats'):
                continue
            
            stats = data['response']['stats'][:10]  # Top 10 for each category
            
            for i, stat in enumerate(stats):
                player_name = stat.get('player', {}).get('title', 'Unknown')
                team_name = stat.get('team', {}).get('title', 'Unknown')
                
                # Top Wicket Takers
                if "top_wicket_takers" in filename:
                    wickets = stat.get('wickets', 0)
                    matches = stat.get('matches', 0)
                    runs = stat.get('runs', 0)
                    avg = stat.get('average', 0)
                    economy = stat.get('econ', 0)
                    
                    if i == 0:
                        qa_pairs.append((
                            f"Who took the most wickets in IPL 2022?",
                            f"{player_name} from {team_name} took the most wickets with {wickets} wickets in {matches} matches, conceding {runs} runs at an average of {avg} and economy of {economy}."
                        ))
                    
                    qa_pairs.append((
                        f"How many wickets did {player_name} take in IPL 2022?",
                        f"{player_name} took {wickets} wickets in {matches} matches for {team_name}, with a bowling average of {avg}."
                    ))
                    
                    qa_pairs.append((
                        f"What is {player_name}'s bowling average in IPL 2022?",
                        f"{player_name}'s bowling average in IPL 2022 is {avg}. He took {wickets} wickets in {matches} matches."
                    ))
                
                # Best Economy Rates
                elif "best_economy_rates.json" in filename and "innings" not in filename:
                    economy = stat.get('econ', 0)
                    wickets = stat.get('wickets', 0)
                    runs = stat.get('runs', 0)
                    overs = stat.get('overs', 0)
                    
                    if i == 0:
                        qa_pairs.append((
                            f"Which bowler has the best economy rate in IPL 2022?",
                            f"{player_name} from {team_name} has the best economy rate of {economy}, taking {wickets} wickets in {overs} overs while conceding {runs} runs."
                        ))
                    
                    qa_pairs.append((
                        f"What is {player_name}'s economy rate in IPL 2022?",
                        f"{player_name} has an economy rate of {economy} for {team_name}, conceding {runs} runs in {overs} overs."
                    ))
                
                # Best Bowling Averages
                elif "best_averages.json" in filename:
                    avg = stat.get('average', 0)
                    wickets = stat.get('wickets', 0)
                    runs = stat.get('runs', 0)
                    matches = stat.get('matches', 0)
                    
                    if i == 0:
                        qa_pairs.append((
                            f"Which bowler has the best bowling average in IPL 2022?",
                            f"{player_name} from {team_name} has the best bowling average of {avg}, taking {wickets} wickets while conceding {runs} runs in {matches} matches."
                        ))
                    
                    qa_pairs.append((
                        f"What is the bowling average of {player_name} in IPL 2022?",
                        f"{player_name} has a bowling average of {avg}, taking {wickets} wickets in {matches} matches for {team_name}."
                    ))
                
                # Best Bowling Figures
                elif "best_bowling_figures" in filename:
                    best = stat.get('bestinning', '0/0')
                    wickets = stat.get('wickets', 0)
                    matches = stat.get('matches', 0)
                    
                    if i == 0:
                        qa_pairs.append((
                            f"What are the best bowling figures in IPL 2022?",
                            f"{player_name} from {team_name} has the best bowling figures of {best} in a single match, with {wickets} total wickets in the season."
                        ))
                    
                    qa_pairs.append((
                        f"What are {player_name}'s best bowling figures in IPL 2022?",
                        f"{player_name}'s best bowling figures are {best} in a single match. He took a total of {wickets} wickets in {matches} matches for {team_name}."
                    ))
                
                # Best Strike Rates
                elif "best_strike_rates.json" in filename and "innings" not in filename:
                    strike_rate = stat.get('strike', 0)
                    wickets = stat.get('wickets', 0)
                    balls = stat.get('balls', 0)
                    
                    if i == 0:
                        qa_pairs.append((
                            f"Which bowler has the best strike rate in IPL 2022?",
                            f"{player_name} from {team_name} has the best strike rate of {strike_rate}, taking {wickets} wickets in {balls} balls."
                        ))
                    
                    qa_pairs.append((
                        f"What is {player_name}'s bowling strike rate in IPL 2022?",
                        f"{player_name} has a bowling strike rate of {strike_rate} for {team_name}, taking {wickets} wickets in {balls} balls."
                    ))
                
                # Five-Wicket Hauls
                elif "five_wickets" in filename:
                    five_wickets = stat.get('wickets5', 0)
                    wickets = stat.get('wickets', 0)
                    matches = stat.get('matches', 0)
                    
                    if five_wickets > 0:
                        qa_pairs.append((
                            f"How many five-wicket hauls did {player_name} take in IPL 2022?",
                            f"{player_name} from {team_name} took {five_wickets} five-wicket haul(s) in IPL 2022, with {wickets} total wickets in {matches} matches."
                        ))
                        
                        if i == 0:
                            qa_pairs.append((
                                f"Who took the most five-wicket hauls in IPL 2022?",
                                f"{player_name} from {team_name} took {five_wickets} five-wicket haul(s) in IPL 2022."
                            ))
                
                # Four-Wicket Hauls
                elif "four_wickets" in filename:
                    four_wickets = stat.get('wickets4', 0)
                    wickets = stat.get('wickets', 0)
                    matches = stat.get('matches', 0)
                    
                    if four_wickets > 0:
                        qa_pairs.append((
                            f"How many four-wicket hauls did {player_name} take in IPL 2022?",
                            f"{player_name} from {team_name} took {four_wickets} four-wicket haul(s) in IPL 2022, with {wickets} total wickets in {matches} matches."
                        ))
                
                # Maiden Overs
                elif "maidens" in filename:
                    maidens = stat.get('maidens', 0)
                    overs = stat.get('overs', 0)
                    wickets = stat.get('wickets', 0)
                    
                    if maidens > 0:
                        qa_pairs.append((
                            f"How many maiden overs did {player_name} bowl in IPL 2022?",
                            f"{player_name} from {team_name} bowled {maidens} maiden over(s) in {overs} overs, taking {wickets} wickets."
                        ))
                
                # Innings-specific records
                elif "innings" in filename:
                    if "economy_rates_innings" in filename:
                        economy = stat.get('econ', 0)
                        runs = stat.get('runs', 0)
                        overs = stat.get('overs', 0)
                        wickets = stat.get('wickets', 0)
                        
                        qa_pairs.append((
                            f"What is {player_name}'s best economy in an innings in IPL 2022?",
                            f"{player_name} achieved an economy rate of {economy} in an innings, conceding {runs} runs in {overs} overs and taking {wickets} wicket(s) for {team_name}."
                        ))
                    elif "strike_rates_innings" in filename:
                        strike = stat.get('strike', 0)
                        wickets = stat.get('wickets', 0)
                        balls = stat.get('balls', 0)
                        
                        qa_pairs.append((
                            f"What is {player_name}'s best strike rate in an innings in IPL 2022?",
                            f"{player_name} achieved a strike rate of {strike} in an innings, taking {wickets} wicket(s) in {balls} balls."
                        ))
                    elif "most_runs_conceded" in filename:
                        runs = stat.get('runs', 0)
                        overs = stat.get('overs', 0)
                        wickets = stat.get('wickets', 0)
                        
                        qa_pairs.append((
                            f"What is the most runs {player_name} conceded in an innings in IPL 2022?",
                            f"{player_name} conceded {runs} runs in {overs} overs while taking {wickets} wicket(s) in an innings."
                        ))
        
        return qa_pairs
    
    
    def generate_match_info_qa(self) -> List[Tuple[str, str]]:
        """Generate Q&A pairs from match info files."""
        qa_pairs = []
        
        if not self.match_info_dir.exists():
            return qa_pairs
        
        match_files = list(self.match_info_dir.glob("*.json"))
        logger.info(f"Processing {len(match_files)} match info files...")
        
        for match_file in match_files:
            data = self.load_json_file(match_file)
            
            # Match info files are flat JSON, not wrapped in response
            if not isinstance(data, dict) or not data.get('title'):
                continue
            
            match_data = data
            
            # Extract match details
            match_title = match_data.get('title', 'Unknown Match')
            venue = match_data.get('venue', {}).get('name', 'Unknown Venue')
            city = match_data.get('venue', {}).get('location', 'Unknown City')
            result = match_data.get('status_note', match_data.get('result', 'No result'))
            
            # Get team names from teama and teamb
            teama = match_data.get('teama', {})
            teamb = match_data.get('teamb', {})
            team1 = teama.get('name', 'Team 1')
            team2 = teamb.get('name', 'Team 2')
            
            # Get winner
            winning_team_id = match_data.get('winning_team_id')
            winner = team1 if winning_team_id == teama.get('team_id') else team2
            
            # Match result questions
            qa_pairs.append((
                f"Who won the match between {team1} and {team2} in IPL 2022?",
                f"{winner} won the match at {venue}, {city}. {result}"
            ))
            
            # Venue questions
            qa_pairs.append((
                f"Where was the match between {team1} and {team2} played in IPL 2022?",
                f"The match was played at {venue} in {city}."
            ))
            
            # Toss questions
            toss_data = match_data.get('toss', {})
            if toss_data:
                toss_text = toss_data.get('text', '')
                if toss_text:
                    qa_pairs.append((
                        f"What was the toss decision in the match between {team1} and {team2}?",
                        f"{toss_text}."
                    ))
            
            # Man of the match
            mom = match_data.get('man_of_the_match', {})
            if mom:
                mom_name = mom.get('name')
                if mom_name:
                    qa_pairs.append((
                        f"Who was the man of the match in {team1} vs {team2}?",
                        f"{mom_name} was awarded Man of the Match."
                    ))
        
        return qa_pairs
    
    def generate_scorecard_qa(self) -> List[Tuple[str, str]]:
        """Generate Q&A pairs from scorecard files."""
        qa_pairs = []
        
        if not self.scorecards_dir.exists():
            return qa_pairs
        
        scorecard_files = list(self.scorecards_dir.glob("*.json"))
        logger.info(f"Processing {len(scorecard_files)} scorecard files...")
        
        for scorecard_file in scorecard_files[:50]:  # Process first 50 to avoid excessive data
            data = self.load_json_file(scorecard_file)
            
            # Scorecard files are flat JSON
            if not isinstance(data, dict) or not data.get('innings'):
                continue
            
            card = data
            match_title = card.get('title', 'Unknown Match')
            
            innings = card.get('innings', [])
            for inning in innings:
                batting_team_id = inning.get('batting_team_id')
                # Get team name from teama or teamb
                if batting_team_id == card.get('teama', {}).get('team_id'):
                    batting_team = card.get('teama', {}).get('name', 'Unknown Team')
                else:
                    batting_team = card.get('teamb', {}).get('name', 'Unknown Team')
                
                equations = inning.get('equations', {})
                total_runs = equations.get('runs', 0)
                wickets = equations.get('wickets', 0)
                overs = equations.get('overs', '0')
                
                # Team score questions
                qa_pairs.append((
                    f"What was {batting_team}'s score in {match_title}?",
                    f"{batting_team} scored {total_runs}/{wickets} in {overs} overs."
                ))
                
                # Top scorers from innings
                batsmen = inning.get('batsmen', [])
                if batsmen:
                    top_scorer = batsmen[0]
                    batsman_name = top_scorer.get('name', 'Unknown')
                    runs = top_scorer.get('runs', 0)
                    balls = top_scorer.get('balls_faced', 0)
                    
                    qa_pairs.append((
                        f"Who was the top scorer for {batting_team} in {match_title}?",
                        f"{batsman_name} was the top scorer with {runs} runs off {balls} balls for {batting_team}."
                    ))
        
        return qa_pairs
    
    def generate_team_stats_qa(self) -> List[Tuple[str, str]]:
        """Generate Q&A pairs from team statistics files."""
        qa_pairs = []
        
        if not self.team_stats_dir.exists():
            return qa_pairs
        
        team_files = list(self.team_stats_dir.glob("*.json"))
        logger.info(f"Processing {len(team_files)} team stats files...")
        
        for team_file in team_files:
            data = self.load_json_file(team_file)
            
            # Team stats files are wrapped: {response: {stats: [{score, team}, ...]}}
            if not isinstance(data, dict):
                continue
            
            stats = self.extract_data(data, 'stats')
            if not stats:
                stats = data.get('stats', [])
            
            if not stats or not isinstance(stats, list):
                continue
            
            # Extract stat type from filename
            stat_type = team_file.stem.replace('team_', '').replace('_', ' ').title()
            
            # Process team stats based on file type
            for i, stat in enumerate(stats[:10]):  # Top 10
                if not isinstance(stat, dict):
                    continue
                    
                score = stat.get('score', 'Unknown')
                team_info = stat.get('team', {})
                team_name = team_info.get('title', 'Unknown')
                
                if 'highest_score' in team_file.name:
                    if i == 0:
                        qa_pairs.append((
                            f"What is the highest score by any team in IPL 2022?",
                            f"{team_name} scored {score} which is the highest score in IPL 2022."
                        ))
                    qa_pairs.append((
                        f"Did {team_name} score {score} in IPL 2022?",
                        f"Yes, {team_name} scored {score} in IPL 2022."
                    ))
                
                elif 'lowest_score' in team_file.name:
                    if i == 0:
                        qa_pairs.append((
                            f"What is the lowest score by any team in IPL 2022?",
                            f"{team_name} scored {score} which is the lowest complete innings score in IPL 2022."
                        ))
                
                elif 'win_margin' in team_file.name:
                    margin = stat.get('margin', score)
                    if 'runs' in team_file.name:
                        qa_pairs.append((
                            f"What was {team_name}'s biggest win margin by runs in IPL 2022?",
                            f"{team_name} won by {margin} runs in IPL 2022."
                        ))
                    elif 'wickets' in team_file.name:
                        qa_pairs.append((
                            f"What was {team_name}'s biggest win margin by wickets in IPL 2022?",
                            f"{team_name} won by {margin} wickets in IPL 2022."
                        ))
                
                elif 'extra_run' in team_file.name:
                    extras = stat.get('extras', score)
                    qa_pairs.append((
                        f"How many extra runs did {team_name} concede in a match in IPL 2022?",
                        f"{team_name} conceded {extras} extra runs in a match."
                    ))
        
        return qa_pairs
    
    def generate_standings_qa(self) -> List[Tuple[str, str]]:
        """Generate Q&A pairs from standings/points table files."""
        qa_pairs = []
        
        if not self.standings_dir.exists():
            return qa_pairs
        
        standings_files = list(self.standings_dir.glob("*.json"))
        logger.info(f"Processing {len(standings_files)} standings files...")
        
        for standings_file in standings_files:
            data = self.load_json_file(standings_file)
            
            # Handle both wrapped and flat JSON formats
            standings = self.extract_data(data, 'standings')
            if not standings:
                standings = data.get('standings', [])
            
            if standings:
                
                for i, team_standing in enumerate(standings[:8]):  # Top 8 teams
                    team_name = team_standing.get('team', {}).get('title', 'Unknown')
                    position = team_standing.get('position', 0)
                    played = team_standing.get('played', 0)
                    won = team_standing.get('won', 0)
                    lost = team_standing.get('lost', 0)
                    points = team_standing.get('points', 0)
                    nrr = team_standing.get('nrr', 0)
                    
                    if i == 0:
                        qa_pairs.append((
                            f"Which team finished at the top of the points table in IPL 2022?",
                            f"{team_name} finished at position {position} with {points} points, winning {won} out of {played} matches with a net run rate of {nrr}."
                        ))
                    
                    qa_pairs.append((
                        f"What position did {team_name} finish in the IPL 2022 points table?",
                        f"{team_name} finished at position {position} with {points} points. They won {won} and lost {lost} out of {played} matches, with a net run rate of {nrr}."
                    ))
                    
                    qa_pairs.append((
                        f"How many points did {team_name} earn in IPL 2022?",
                        f"{team_name} earned {points} points by winning {won} matches out of {played} played."
                    ))
        
        return qa_pairs
    
    def generate_squads_qa(self) -> List[Tuple[str, str]]:
        """Generate Q&A pairs from squad files."""
        qa_pairs = []
        
        if not self.squads_dir.exists():
            return qa_pairs
        
        squad_files = list(self.squads_dir.glob("*.json"))
        logger.info(f"Processing {len(squad_files)} squad files...")
        
        for squad_file in squad_files:
            data = self.load_json_file(squad_file)
            
            # Handle both list and dict formats
            players = None
            team_name = squad_file.stem.replace('_', ' ')
            
            if isinstance(data, list):
                # Direct list of players
                players = data
            elif isinstance(data, dict) and data.get('response', {}).get('players'):
                # Dict with response.players
                players = data['response']['players']
            elif isinstance(data, dict) and data.get('players'):
                # Dict with direct players key
                players = data['players']
            
            if players:
                # Count players by role
                batsmen = []
                bowlers = []
                allrounders = []
                wicketkeepers = []
                
                for player in players:
                    if not isinstance(player, dict):
                        continue
                        
                    player_name = player.get('title', player.get('name', 'Unknown'))
                    role = player.get('playing_role', player.get('role', '')).lower()
                    
                    if 'bat' in role and 'bowl' not in role:
                        batsmen.append(player_name)
                    elif 'bowl' in role and 'bat' not in role:
                        bowlers.append(player_name)
                    elif 'allrounder' in role or ('bat' in role and 'bowl' in role):
                        allrounders.append(player_name)
                    elif 'keeper' in role or 'wk' in role:
                        wicketkeepers.append(player_name)
                
                # Squad composition questions
                qa_pairs.append((
                    f"How many players are in {team_name} squad for IPL 2022?",
                    f"{team_name} has {len(players)} players in their IPL 2022 squad."
                ))
                
                if batsmen:
                    qa_pairs.append((
                        f"Who are the batsmen in {team_name} squad for IPL 2022?",
                        f"{team_name} has {len(batsmen)} specialist batsmen including {', '.join(batsmen[:5])}."
                    ))
                
                if bowlers:
                    qa_pairs.append((
                        f"Who are the bowlers in {team_name} squad for IPL 2022?",
                        f"{team_name} has {len(bowlers)} specialist bowlers including {', '.join(bowlers[:5])}."
                    ))
                
                if allrounders:
                    qa_pairs.append((
                        f"Who are the all-rounders in {team_name} squad for IPL 2022?",
                        f"{team_name} has {len(allrounders)} all-rounders including {', '.join(allrounders[:3])}."
                    ))
        
        return qa_pairs
    
    def generate_teams_qa(self) -> List[Tuple[str, str]]:
        """Generate Q&A pairs from teams files."""
        qa_pairs = []
        
        if not self.teams_dir.exists():
            return qa_pairs
        
        team_files = list(self.teams_dir.glob("*.json"))
        logger.info(f"Processing {len(team_files)} team files...")
        
        for team_file in team_files:
            data = self.load_json_file(team_file)
            
            # teams.json is a direct list, not wrapped in dict
            if isinstance(data, list):
                teams = data
            elif isinstance(data, dict):
                # Extract teams array from response or direct
                teams = self.extract_data(data, 'teams')
                if not teams:
                    teams = data.get('teams', [])
            else:
                continue
            
            if isinstance(teams, list) and len(teams) > 0:
                # Generate a question about total IPL teams
                team_names = [t.get('title', t.get('name', 'Unknown')) for t in teams]
                qa_pairs.append((
                    f"How many teams participated in IPL 2022?",
                    f"{len(teams)} teams participated in IPL 2022: {', '.join(team_names)}."
                ))
                
                # Process individual teams
                for team in teams:
                    team_name = team.get('title', team.get('name', 'Unknown'))
                    abbr = team.get('abbr', team.get('short_name', ''))
                    
                    if abbr and team_name != 'Unknown':
                        qa_pairs.append((
                            f"What is the full name of {abbr} in IPL 2022?",
                            f"{abbr} stands for {team_name} in IPL 2022."
                        ))
        
        return qa_pairs
    
    def generate_player_career_qa(self) -> List[Tuple[str, str]]:
        """Generate Q&A pairs from player career stats files."""
        qa_pairs = []
        
        if not self.player_career_dir.exists():
            return qa_pairs
        
        player_files = list(self.player_career_dir.glob("*.json"))
        logger.info(f"Processing {len(player_files)} player career files...")
        
        for player_file in player_files[:50]:  # Process first 50 to avoid excessive data
            data = self.load_json_file(player_file)
            
            if not isinstance(data, dict):
                continue
            
            # Get player info
            player_data = data.get('player', {})
            if not isinstance(player_data, dict):
                continue
            
            player_name = player_data.get('title', player_data.get('name', 'Unknown'))
            if player_name == 'Unknown':
                continue
            
            # Batting career stats (t20 format for IPL)
            batting_data = data.get('batting', {})
            if isinstance(batting_data, dict):
                t20_batting = batting_data.get('t20', {})
                if isinstance(t20_batting, dict) and t20_batting:
                    matches = int(t20_batting.get('matches', 0) or 0)
                    runs = int(t20_batting.get('runs', 0) or 0)
                    
                    if matches > 50 and runs > 500:  # Only significant T20 careers
                        avg = t20_batting.get('average', '0')
                        strike = t20_batting.get('strike', '0')
                        hundreds = int(t20_batting.get('run100', 0) or 0)
                        fifties = int(t20_batting.get('run50', 0) or 0)
                        
                        qa_pairs.append((
                            f"What are {player_name}'s T20 career batting stats?",
                            f"{player_name} has played {matches} T20 matches, scoring {runs} runs with an average of {avg} and strike rate of {strike}."
                        ))
                        
                        if hundreds > 0 or fifties > 5:
                            qa_pairs.append((
                                f"How many centuries and fifties has {player_name} scored in T20?",
                                f"{player_name} has scored {hundreds} centuries and {fifties} fifties in T20 cricket."
                            ))
            
            # Bowling career stats (t20 format for IPL)
            bowling_data = data.get('bowling', {})
            if isinstance(bowling_data, dict):
                t20_bowling = bowling_data.get('t20', {})
                if isinstance(t20_bowling, dict) and t20_bowling:
                    matches = int(t20_bowling.get('matches', 0) or 0)
                    wickets = int(t20_bowling.get('wickets', 0) or 0)
                    
                    if wickets >= 50:  # Only notable T20 bowlers
                        economy = t20_bowling.get('econ', '0')
                        average = t20_bowling.get('average', '0')
                        
                        qa_pairs.append((
                            f"How many wickets has {player_name} taken in T20?",
                            f"{player_name} has taken {wickets} wickets in {matches} T20 matches with an economy of {economy} and average of {average}."
                        ))
        
        return qa_pairs
    
    def generate_matches_qa(self) -> List[Tuple[str, str]]:
        """Generate Q&A pairs from matches directory files."""
        qa_pairs = []
        
        if not self.matches_dir.exists():
            return qa_pairs
        
        match_files = list(self.matches_dir.glob("*.json"))
        logger.info(f"Processing {len(match_files)} match files...")
        
        for match_file in match_files:
            data = self.load_json_file(match_file)
            
            if not isinstance(data, list):
                continue
            
            # matches.json contains a direct array of match objects
            total_matches = len(data)
            
            # Generate overview question
            if total_matches > 0:
                qa_pairs.append((
                    f"How many matches were played in IPL 2022?",
                    f"{total_matches} matches were played in IPL 2022 season."
                ))
            
            # Process sample matches
            for match in data[:20]:  # Process first 20 matches
                if not isinstance(match, dict):
                    continue
                
                match_title = match.get('title', match.get('short_title', 'Unknown'))
                match_number = match.get('match_number', 'Unknown')
                status_note = match.get('status_note', '')
                venue_data = match.get('venue', {})
                venue_name = venue_data.get('name', venue_data.get('location', 'Unknown'))
                
                if match_title != 'Unknown' and status_note:
                    qa_pairs.append((
                        f"What was the result of Match {match_number} in IPL 2022?",
                        f"{match_title} was played at {venue_name}. {status_note}."
                    ))
        
        return qa_pairs
    
    def generate_player_comparison_qa(self) -> List[Tuple[str, str]]:
        """Generate player comparison and team-based Q&A pairs."""
        qa_pairs = []
        
        # Load top batsmen
        most_runs_file = self.batting_stats_dir / "batting_most_runs.json"
        data = self.load_json_file(most_runs_file)
        
        if data.get('response', {}).get('stats'):
            all_stats = data['response']['stats'][:15]
            
            # Team-based questions
            teams_data = {}
            for stat in all_stats:
                team = stat.get('team', {}).get('title', 'Unknown')
                if team not in teams_data:
                    teams_data[team] = []
                teams_data[team].append(stat)
            
            for team, players in teams_data.items():
                if len(players) > 0:
                    top_player = players[0]
                    player_name = top_player.get('player', {}).get('title', 'Unknown')
                    runs = top_player.get('runs', 0)
                    
                    qa_pairs.append((
                        f"Who is the top run scorer for {team} in IPL 2022?",
                        f"{player_name} is the leading run-scorer for {team} in IPL 2022 with {runs} runs."
                    ))
            
            # Comparison questions
            if len(all_stats) >= 2:
                player1 = all_stats[0]
                player2 = all_stats[1]
                name1 = player1.get('player', {}).get('title', 'Player 1')
                name2 = player2.get('player', {}).get('title', 'Player 2')
                runs1 = player1.get('runs', 0)
                runs2 = player2.get('runs', 0)
                
                qa_pairs.append((
                    f"Who scored more runs in IPL 2022, {name1} or {name2}?",
                    f"{name1} scored {runs1} runs while {name2} scored {runs2} runs, so {name1} scored more."
                ))
        
        # Load top wicket takers
        wickets_file = self.bowling_stats_dir / "bowling_top_wicket_takers.json"
        data = self.load_json_file(wickets_file)
        
        if data.get('response', {}).get('stats'):
            all_stats = data['response']['stats'][:15]
            
            # Team-based bowling questions
            teams_data = {}
            for stat in all_stats:
                team = stat.get('team', {}).get('title', 'Unknown')
                if team not in teams_data:
                    teams_data[team] = []
                teams_data[team].append(stat)
            
            for team, players in teams_data.items():
                if len(players) > 0:
                    top_player = players[0]
                    player_name = top_player.get('player', {}).get('title', 'Unknown')
                    wickets = top_player.get('wickets', 0)
                    
                    qa_pairs.append((
                        f"Who is the top wicket taker for {team} in IPL 2022?",
                        f"{player_name} is the leading wicket-taker for {team} in IPL 2022 with {wickets} wickets."
                    ))
        
        return qa_pairs
    
    def generate_match_commentary_qa(self) -> List[Tuple[str, str]]:
        """Generate Q&A pairs from match innings commentary files."""
        qa_pairs = []
        
        if not self.match_commentary_dir.exists():
            return qa_pairs
        
        commentary_files = list(self.match_commentary_dir.glob("*.json"))
        logger.info(f"Processing {len(commentary_files)} match commentary files...")
        
        # Commentary files are very large - process fewer
        for commentary_file in commentary_files[:10]:  # Process only first 10
            data = self.load_json_file(commentary_file)
            
            if not isinstance(data, dict):
                continue
            
            # Extract teams from commentary file
            teams = data.get('teams', [])
            if not teams or len(teams) < 2:
                continue
            
            team1_name = teams[0].get('name', 'Team 1')
            team2_name = teams[1].get('name', 'Team 2')
            
            # Extract match title from filename
            match_desc = commentary_file.stem.replace('innings_1_', '').replace('innings_2_', '').replace('_commentary', '').replace('_', ' ')
            
            # Get commentaries array
            commentaries = data.get('commentaries', [])
            if isinstance(commentaries, list) and len(commentaries) > 0:
                # Count significant events
                wicket_count = sum(1 for c in commentaries if isinstance(c, dict) and 'wicket' in str(c.get('event', '')).lower())
                boundary_count = sum(1 for c in commentaries if isinstance(c, dict) and c.get('run', 0) in [4, 6])
                
                if wicket_count > 0:
                    qa_pairs.append((
                        f"How many wickets fell in the {match_desc} innings?",
                        f"{wicket_count} wickets fell during this innings between {team1_name} and {team2_name}."
                    ))
                
                if boundary_count > 5:
                    qa_pairs.append((
                        f"How many boundaries were hit in the {match_desc}?",
                        f"{boundary_count} boundaries (fours and sixes) were hit in this innings."
                    ))
        
        return qa_pairs
    
    def generate_match_live_details_qa(self) -> List[Tuple[str, str]]:
        """Generate Q&A pairs from match live details files."""
        qa_pairs = []
        
        if not self.match_live_dir.exists():
            return qa_pairs
        
        live_files = list(self.match_live_dir.glob("*.json"))
        logger.info(f"Processing {len(live_files)} match live details files...")
        
        # Process sample live details files
        for live_file in live_files[:15]:  # Process first 15
            data = self.load_json_file(live_file)
            
            if not isinstance(data, dict):
                continue
            
            # Extract teams
            teams = data.get('teams', [])
            if not teams or len(teams) < 2:
                continue
            
            team1 = teams[0]
            team2 = teams[1]
            
            team1_name = team1.get('name', 'Team 1')
            team2_name = team2.get('name', 'Team 2')
            
            # Get match status
            status_note = data.get('status_note', '')
            
            if status_note:
                qa_pairs.append((
                    f"What was the live match status for {team1_name} vs {team2_name}?",
                    f"{status_note}"
                ))
        
        return qa_pairs
    
    def generate_wagon_wheel_qa(self) -> List[Tuple[str, str]]:
        """Generate Q&A pairs from wagon wheel (shot placement) files."""
        qa_pairs = []
        
        if not self.match_wagon_dir.exists():
            return qa_pairs
        
        wagon_files = list(self.match_wagon_dir.glob("*.json"))
        logger.info(f"Processing {len(wagon_files)} wagon wheel files...")
        
        # Process a sample of wagon wheel files
        for wagon_file in wagon_files[:15]:  # Process 15 files
            data = self.load_json_file(wagon_file)
            
            if not isinstance(data, dict):
                continue
            
            # Extract innings data
            innings = data.get('innings', [])
            if not isinstance(innings, list) or len(innings) == 0:
                continue
            
            # Get teams for context
            teams = data.get('teams', [])
            team_map = {t.get('tid'): t.get('title', 'Unknown') for t in teams if isinstance(t, dict)}
            
            for inning in innings[:2]:  # Both innings
                if not isinstance(inning, dict):
                    continue
                
                team_id = inning.get('batting_team_id')
                team_name = team_map.get(team_id, inning.get('name', 'Unknown Team'))
                runs = inning.get('runs', 0)
                wickets = inning.get('wickets', 0)
                overs = inning.get('overs', '0')
                
                # Count boundaries from batsmen data
                batsmen = inning.get('batsmen', [])
                total_fours = sum(b.get('fours', 0) for b in batsmen if isinstance(b, dict))
                total_sixes = sum(b.get('sixes', 0) for b in batsmen if isinstance(b, dict))
                
                if runs > 0:
                    qa_pairs.append((
                        f"How many boundaries did {team_name} hit in their innings?",
                        f"{team_name} hit {total_fours} fours and {total_sixes} sixes in their innings of {runs}/{wickets} in {overs} overs."
                    ))
                    
                    # Find top scorer
                    if len(batsmen) > 0:
                        top_batsman = max((b for b in batsmen if isinstance(b, dict) and b.get('runs', 0) > 0), 
                                        key=lambda x: x.get('runs', 0), default=None)
                        if top_batsman:
                            player_runs = top_batsman.get('runs', 0)
                            player_balls = top_batsman.get('balls_faced', 0)
                            player_fours = top_batsman.get('fours', 0)
                            player_sixes = top_batsman.get('sixes', 0)
                            
                            # We don't have player names, so use generic question
                            qa_pairs.append((
                                f"What was the highest individual score in {team_name}'s innings?",
                                f"The top scorer made {player_runs} runs off {player_balls} balls with {player_fours} fours and {player_sixes} sixes."
                            ))
                    break  # Only process first innings per file to avoid duplicates
        
        return qa_pairs
    
    def generate_all_qa_pairs(self) -> List[Dict[str, str]]:
        """Generate all Q&A pairs from all categories."""
        all_qa = []
        
        logger.info("=" * 60)
        logger.info("GENERATING COMPREHENSIVE IPL 2022 DATASET")
        logger.info("Processing ALL directories and files")
        logger.info("=" * 60)
        
        # Batting statistics (11 files)
        logger.info("\n[1/12] Generating batting Q&A pairs...")
        batting_qa = self.generate_batting_qa()
        logger.info(f"✓ Generated {len(batting_qa)} batting Q&A pairs")
        all_qa.extend(batting_qa)
        
        # Bowling statistics (11 files)
        logger.info("\n[2/12] Generating bowling Q&A pairs...")
        bowling_qa = self.generate_bowling_qa()
        logger.info(f"✓ Generated {len(bowling_qa)} bowling Q&A pairs")
        all_qa.extend(bowling_qa)
        
        # Match information
        logger.info("\n[3/12] Generating match info Q&A pairs...")
        match_qa = self.generate_match_info_qa()
        logger.info(f"✓ Generated {len(match_qa)} match info Q&A pairs")
        all_qa.extend(match_qa)
        
        # Scorecards
        logger.info("\n[4/12] Generating scorecard Q&A pairs...")
        scorecard_qa = self.generate_scorecard_qa()
        logger.info(f"✓ Generated {len(scorecard_qa)} scorecard Q&A pairs")
        all_qa.extend(scorecard_qa)
        
        # Team statistics
        logger.info("\n[5/12] Generating team stats Q&A pairs...")
        team_stats_qa = self.generate_team_stats_qa()
        logger.info(f"✓ Generated {len(team_stats_qa)} team stats Q&A pairs")
        all_qa.extend(team_stats_qa)
        
        # Standings/Points table
        logger.info("\n[6/12] Generating standings Q&A pairs...")
        standings_qa = self.generate_standings_qa()
        logger.info(f"✓ Generated {len(standings_qa)} standings Q&A pairs")
        all_qa.extend(standings_qa)
        
        # Squads
        logger.info("\n[7/12] Generating squad Q&A pairs...")
        squads_qa = self.generate_squads_qa()
        logger.info(f"✓ Generated {len(squads_qa)} squad Q&A pairs")
        all_qa.extend(squads_qa)
        
        # Teams info
        logger.info("\n[8/12] Generating teams Q&A pairs...")
        teams_qa = self.generate_teams_qa()
        logger.info(f"✓ Generated {len(teams_qa)} teams Q&A pairs")
        all_qa.extend(teams_qa)
        
        # Player career stats
        logger.info("\n[9/12] Generating player career Q&A pairs...")
        player_career_qa = self.generate_player_career_qa()
        logger.info(f"✓ Generated {len(player_career_qa)} player career Q&A pairs")
        all_qa.extend(player_career_qa)
        
        # Match innings commentary
        logger.info("\n[10/12] Generating match commentary Q&A pairs...")
        commentary_qa = self.generate_match_commentary_qa()
        logger.info(f"✓ Generated {len(commentary_qa)} commentary Q&A pairs")
        all_qa.extend(commentary_qa)
        
        # Match live details
        logger.info("\n[11/12] Generating match live details Q&A pairs...")
        live_qa = self.generate_match_live_details_qa()
        logger.info(f"✓ Generated {len(live_qa)} live details Q&A pairs")
        all_qa.extend(live_qa)
        
        # Wagon wheel data
        logger.info("\n[12/12] Generating wagon wheel Q&A pairs...")
        wagon_qa = self.generate_wagon_wheel_qa()
        logger.info(f"✓ Generated {len(wagon_qa)} wagon wheel Q&A pairs")
        all_qa.extend(wagon_qa)
        
        # Comparison and aggregate questions
        logger.info("\n[BONUS] Generating comparison Q&A pairs...")
        comparison_qa = self.generate_player_comparison_qa()
        logger.info(f"✓ Generated {len(comparison_qa)} comparison Q&A pairs")
        all_qa.extend(comparison_qa)
        
        # Matches
        logger.info("\n[BONUS] Generating matches Q&A pairs...")
        matches_qa = self.generate_matches_qa()
        logger.info(f"✓ Generated {len(matches_qa)} matches Q&A pairs")
        all_qa.extend(matches_qa)
        
        logger.info("\n" + "=" * 60)
        logger.info(f"TOTAL Q&A pairs before deduplication: {len(all_qa)}")
        logger.info("=" * 60)
        
        # Remove duplicates based on questions
        seen_questions = set()
        unique_qa = []
        for question, answer in all_qa:
            if question not in seen_questions:
                seen_questions.add(question)
                unique_qa.append({"question": question, "answer": answer})
        
        logger.info(f"\n✓ Total UNIQUE Q&A pairs: {len(unique_qa)}")
        logger.info("=" * 60)
        
        return unique_qa
    
    def save_dataset(self, output_path: str):
        """Save the generated dataset to a JSON file."""
        dataset = self.generate_all_qa_pairs()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Dataset saved to {output_path}")
        logger.info(f"Total training examples: {len(dataset)}")
        
        # Print sample
        logger.info("\nSample Q&A pairs:")
        for i, pair in enumerate(dataset[:5]):
            logger.info(f"\nQ{i+1}: {pair['question']}")
            logger.info(f"A{i+1}: {pair['answer']}")


def main():
    """Main function to generate IPL dataset."""
    # Path to the IPL data directory
    ipl_data_dir = "data/Indian_Premier_League_2022-03-26"
    output_path = "data/ipl_qa.json"
    
    if not os.path.exists(ipl_data_dir):
        logger.error(f"IPL data directory not found at {ipl_data_dir}")
        return
    
    generator = IPLDatasetGenerator(ipl_data_dir)
    generator.save_dataset(output_path)
    
    logger.info("\n✓ Dataset generation complete!")


if __name__ == "__main__":
    main()
