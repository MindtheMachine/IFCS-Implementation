"""
Gradio Web Interface for Trilogy System
Deployable to Replit
"""

import gradio as gr
import json
import os
from typing import Optional, Dict, List, Tuple
import time

from trilogy_app import TrilogyApp
from trilogy_config import TrilogyConfig, TEST_CASES_36_TAXONOMY


class TrilogyWebApp:
    """Web interface for Trilogy system"""

    def __init__(self):
        """Initialize web app"""
        self.app_version = time.strftime("%Y-%m-%d %H:%M:%S")
        self.run_counter = 0
        self.app: Optional[TrilogyApp] = None
        self.config: Optional[TrilogyConfig] = None

        # Load .env file if it exists
        self._load_env_file()

        # Try to load existing configuration from .env
        self.env_provider = os.getenv("LLM_PROVIDER", "").lower()
        self.env_model = os.getenv("LLM_MODEL", "")
        self.env_api_key = os.getenv("LLM_API_KEY", "") or os.getenv("ANTHROPIC_API_KEY", "")

        # Detect provider name for display
        provider_names = {
            "anthropic": "Anthropic Claude",
            "openai": "OpenAI GPT",
            "huggingface": "HuggingFace",
            "ollama": "Ollama (Local)"
        }
        self.provider_display = provider_names.get(self.env_provider, "Not configured")

    def _load_env_file(self):
        """Load .env file if it exists"""
        env_path = os.path.join(os.getcwd(), ".env")
        if os.path.exists(env_path):
            try:
                with open(env_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip()
                            if key and value and not value.startswith('your-'):
                                os.environ[key] = value
            except (UnicodeDecodeError, IOError) as e:
                print(f"Warning: Could not read .env file: {e}")
                print("Will use environment variables or manual configuration")
    
    def initialize_system(
        self,
        api_key: str,
        # ECR params
        ecr_K: int,
        ecr_H: int,
        ecr_tau_CCI: float,
        # Control Probe params
        cp_tau: float,
        cp_Theta: float,
        # IFCS params
        ifcs_rho: float,
        ifcs_lambda_e: float,
        ifcs_lambda_s: float,
        ifcs_lambda_a: float,
    ) -> str:
        """Initialize the trilogy system with configuration

        Returns:
            Status message
        """
        try:
            # Use API key from input if provided, otherwise use from .env
            effective_api_key = api_key.strip() if api_key else self.env_api_key

            if not effective_api_key or effective_api_key.startswith('your-'):
                return "❌ Error: No API key configured. Please either:\n1. Enter API key above, or\n2. Configure .env file with your provider settings"

            # Create config
            config = TrilogyConfig(api_key=effective_api_key)
            
            # Update ECR config
            config.ecr.K = ecr_K
            config.ecr.H = ecr_H
            config.ecr.tau_CCI = ecr_tau_CCI
            
            # Normalize ECR weights (equal for simplicity)
            total = 5.0
            config.ecr.alpha = 1.0 / total
            config.ecr.beta = 1.0 / total
            config.ecr.gamma = 1.0 / total
            config.ecr.delta = 1.0 / total
            config.ecr.epsilon = 1.0 / total
            
            # Update Control Probe config
            config.control_probe.tau = cp_tau
            config.control_probe.Theta = cp_Theta
            
            # Update IFCS config
            config.ifcs.rho = ifcs_rho
            
            # Normalize IFCS weights
            weight_sum = ifcs_lambda_e + ifcs_lambda_s + ifcs_lambda_a
            config.ifcs.lambda_e = ifcs_lambda_e / weight_sum
            config.ifcs.lambda_s = ifcs_lambda_s / weight_sum
            config.ifcs.lambda_a = ifcs_lambda_a / weight_sum
            config.ifcs.lambda_t = 0.0  # Not used for now
            
            # Initialize app
            self.config = config
            self.app = TrilogyApp(config=config)

            # Show which provider is being used
            provider_info = f"Provider: {self.provider_display}" if self.env_provider else f"Provider: {config.provider if hasattr(config, 'provider') else 'Anthropic'}"
            model_info = f"Model: {self.env_model}" if self.env_model else f"Model: {config.model}"

            return f"✅ System initialized successfully!\n\n{provider_info}\n{model_info}"
        
        except Exception as e:
            return f"❌ Initialization failed: {str(e)}"
    
    def process_query(self, prompt: str, context: str = "") -> tuple:
        """Process a query through both baseline and trilogy
        
        Returns:
            (baseline_output, regulated_output, comparison_text, baseline_time, regulated_time, delta_time, status)
        """
        if self.app is None:
            return (
                "❌ System not initialized. Please initialize first.",
                "",
                "",
                gr.update(),
                gr.update(),
                gr.update(),
                "error"
            )
        
        if not prompt.strip():
            return (
                "❌ Please enter a prompt.",
                "",
                "",
                gr.update(),
                gr.update(),
                gr.update(),
                "error"
            )
        
        try:
            self.run_counter += 1
            run_id = f"{self.app_version} / Run {self.run_counter}"

            # Process
            baseline, regulated, comparison = self.app.process_single(prompt, context)
            
            # Save outputs
            self.app.save_outputs(prompt, baseline, regulated, comparison)
            
            # Format comparison
            comparison_text = f"**Run ID:** {run_id}\n\n{self._format_comparison(comparison)}"
            
            baseline_time = comparison.get('baseline_time_s')
            regulated_time = comparison.get('trilogy_time_s')
            delta_time = None
            if baseline_time is not None and regulated_time is not None:
                delta_time = regulated_time - baseline_time
            
            return (
                baseline,
                regulated,
                comparison_text,
                f"{baseline_time:.2f}s" if baseline_time is not None else "",
                f"{regulated_time:.2f}s" if regulated_time is not None else "",
                f"{delta_time:+.2f}s" if delta_time is not None else "",
                "success"
            )
        
        except Exception as e:
            return (
                f"❌ Processing failed: {str(e)}",
                "",
                "",
                gr.update(),
                gr.update(),
                gr.update(),
                "error"
            )
    
    def load_test_case(self, test_case_id: str) -> tuple:
        """Load a test case by ID
        
        Returns:
            (prompt, expected_mechanism)
        """
        # Find test case
        test_case = next((tc for tc in TEST_CASES_36_TAXONOMY if tc['id'] == test_case_id), None)
        
        if test_case:
            prompt = test_case['prompt']
            expected = test_case.get('expected_mechanism_impl', test_case.get('expected_mechanism', 'N/A'))
            expected_paper = test_case.get('expected_mechanism_paper', test_case.get('expected_mechanism', 'N/A'))
            category = test_case['category']
            turns = test_case.get('turns', [])
            
            multi_turn_info = f"\n**Turns:** {len(turns)}" if turns else ""
            info = (
                f"**Test Case:** {test_case_id}\n"
                f"**Category:** {category}\n"
                f"**Expected Mechanism (Impl):** {expected}\n"
                f"**Expected Mechanism (Paper):** {expected_paper}{multi_turn_info}"
            )
            
            return prompt, info
        else:
            return "", "Test case not found"

    def process_test_case(self, test_case_id: str) -> tuple:
        """Process a test case by ID, including multi-turn scenarios.
        
        Returns:
            (baseline_output, regulated_output, comparison_text, baseline_time, regulated_time, delta_time, status)
        """
        if self.app is None:
            return (
                "System not initialized. Please initialize first.",
                "",
                "",
                gr.update(),
                gr.update(),
                gr.update(),
                "error"
            )
        
        test_case = next((tc for tc in TEST_CASES_36_TAXONOMY if tc['id'] == test_case_id), None)
        if not test_case:
            return (
                "Test case not found.",
                "",
                "",
                gr.update(),
                gr.update(),
                gr.update(),
                "error"
            )
        
        if test_case.get('multi_turn') and test_case.get('turns'):
            self.run_counter += 1
            run_id = f"{self.app_version} / Run {self.run_counter}"
            self.app.trilogy.reset_interaction()
            turns = test_case['turns']
            
            baseline_blocks = []
            regulated_blocks = []
            comparison_blocks = []
            
            baseline_time_total = 0.0
            regulated_time_total = 0.0
            
            mechanisms_agg = {
                'ECR': False,
                'CP_Type1': False,
                'IFCS': False,
                'CP_Type2': False
            }
            
            for idx, prompt in enumerate(turns, 1):
                baseline, regulated, comparison = self.app.process_single(prompt, "")
                baseline_time = comparison.get('baseline_time_s') or 0.0
                regulated_time = comparison.get('trilogy_time_s') or 0.0
                baseline_time_total += baseline_time
                regulated_time_total += regulated_time
                
                baseline_blocks.append(f"Turn {idx}:\n{baseline}")
                regulated_blocks.append(f"Turn {idx}:\n{regulated}")
                
                comparison_blocks.append(f"### Turn {idx}\n")
                comparison_blocks.append(self._format_comparison(comparison))
                comparison_blocks.append("")
                
                mech = comparison['mechanisms_fired']
                mechanisms_agg['ECR'] = mechanisms_agg['ECR'] or mech.get('ECR', False)
                mechanisms_agg['CP_Type1'] = mechanisms_agg['CP_Type1'] or mech.get('CP_Type1', False)
                mechanisms_agg['IFCS'] = mechanisms_agg['IFCS'] or mech.get('IFCS', False)
                mechanisms_agg['CP_Type2'] = mechanisms_agg['CP_Type2'] or mech.get('CP_Type2', False)
            
            summary_lines = [
                f"**Run ID:** {run_id}",
                "",
                "## Aggregate Mechanisms Triggered",
                f"- ECR: {'FIRED' if mechanisms_agg['ECR'] else 'Not triggered'}",
                f"- CP_Type1: {'FIRED' if mechanisms_agg['CP_Type1'] else 'Not triggered'}",
                f"- IFCS: {'FIRED' if mechanisms_agg['IFCS'] else 'Not triggered'}",
                f"- CP_Type2: {'FIRED' if mechanisms_agg['CP_Type2'] else 'Not triggered'}",
                ""
            ]
            
            comparison_text = "\n".join(summary_lines + comparison_blocks)
            delta_time = regulated_time_total - baseline_time_total
            
            return (
                "\n\n".join(baseline_blocks),
                "\n\n".join(regulated_blocks),
                comparison_text,
                f"{baseline_time_total:.2f}s",
                f"{regulated_time_total:.2f}s",
                f"{delta_time:+.2f}s",
                "success"
            )
        
        # Single-turn fallback
        return self.process_query(test_case['prompt'], "")
    
    def _format_comparison(self, comparison: Dict) -> str:
        """Format comparison for display"""
        lines = []
        
        # Summary
        lines.append("## 🔍 COMPARISON ANALYSIS\n")
        
        # Mechanisms
        lines.append("### Mechanisms Triggered\n")
        for mechanism, fired in comparison['mechanisms_fired'].items():
            emoji = "✅" if fired else "⭕"
            lines.append(f"{emoji} **{mechanism}**: {'FIRED' if fired else 'Not triggered'}")
        
        lines.append("")
        
        # Admissibility gate details
        cp1 = comparison.get('cp_type1_metrics', {})
        if cp1:
            sigma = cp1.get('sigma')
            sigma_raw = cp1.get('sigma_raw')
            tau = cp1.get('tau')
            prompt_risk = cp1.get('prompt_risk')
            decision = comparison.get('cp_type1_decision', 'N/A').upper()
            
            lines.append("### CP Type-1 Gate Details\n")
            if sigma is not None and tau is not None:
                lines.append(f"- **Decision**: {decision} (σ={sigma:.3f}, τ={tau:.3f})")
            else:
                lines.append(f"- **Decision**: {decision}")
            
            if sigma_raw is not None:
                lines.append(f"- **σ_raw**: {sigma_raw:.3f}")
            if prompt_risk is not None:
                lines.append(f"- **Prompt risk**: {prompt_risk:.2f}")
            lines.append("")
        
        # Key changes
        if comparison.get('key_changes'):
            lines.append("### Key Changes\n")
            for change in comparison['key_changes']:
                lines.append(f"- {change}")
            lines.append("")
        
        # Commitment markers
        markers = comparison.get('commitment_markers', {})
        if markers:
            lines.append("### Commitment Marker Reduction\n")
            baseline = markers.get('baseline', {})
            regulated = markers.get('regulated', {})
            reduction = markers.get('reduction', {})
            
            lines.append(f"- **Universal markers**: {baseline.get('universal', 0)} → {regulated.get('universal', 0)} "
                        f"({reduction.get('universal', 0):+d})")
            lines.append(f"- **Authority markers**: {baseline.get('authority', 0)} → {regulated.get('authority', 0)} "
                        f"({reduction.get('authority', 0):+d})")
            lines.append(f"- **Certainty markers**: {baseline.get('certainty', 0)} → {regulated.get('certainty', 0)} "
                        f"({reduction.get('certainty', 0):+d})")
            lines.append("")
        
        # Statistics
        lines.append("### Statistics\n")
        lines.append(f"- **Response length**: {comparison.get('baseline_length', 0)} → {comparison.get('regulated_length', 0)} chars "
                    f"({comparison.get('length_change_pct', 0):+.1f}%)")
        if comparison.get('baseline_time_s') is not None and comparison.get('trilogy_time_s') is not None:
            delta_time = comparison.get('delta_time_s')
            if delta_time is None:
                delta_time = comparison['trilogy_time_s'] - comparison['baseline_time_s']
            lines.append(f"- **Processing time**: {comparison['baseline_time_s']:.2f}s → {comparison['trilogy_time_s']:.2f}s "
                        f"({delta_time:+.2f}s)")
        else:
            lines.append(f"- **Processing time**: {comparison.get('processing_time_ms', 0):.0f}ms")
        lines.append(f"- **Candidates evaluated**: {comparison.get('num_candidates_evaluated', 0)}")
        
        return "\n".join(lines)


def create_interface():
    """Create Gradio interface"""
    
    web_app = TrilogyWebApp()
    
    with gr.Blocks(title="ECR-Control Probe-IFCS Trilogy System", theme=gr.themes.Soft()) as interface:
        
        gr.Markdown("""
        # 🧠 ECR-Control Probe-IFCS Trilogy System
        
        Implementation of Arijit Chatterjee's inference-time governance framework:
        - **ECR**: Evaluative Coherence Regulation (candidate selection)
        - **Control Probe Type-1**: Admissibility gating
        - **IFCS**: Inference-Time Commitment Shaping
        - **Control Probe Type-2**: Interaction monitoring
        
        ---
        """)
        
        gr.Markdown(f"**UI Build Time:** {web_app.app_version}")
        
        with gr.Tab("🚀 Quick Start"):
            # Show detected configuration
            config_status = "✅ Configuration detected from .env" if web_app.env_api_key else "⚠️ No .env configuration detected"
            provider_status = f"Provider: {web_app.provider_display}" if web_app.env_provider else "Provider: Not configured"
            model_status = f"Model: {web_app.env_model}" if web_app.env_model else "Model: Not configured"

            gr.Markdown(f"### Initialize the system and process queries\n\n**Current Configuration:**\n- {config_status}\n- {provider_status}\n- {model_status}\n")

            with gr.Row():
                api_key_input = gr.Textbox(
                    label="API Key (optional if configured in .env)",
                    placeholder="Leave empty to use .env configuration, or enter API key here",
                    type="password",
                    value=""
                )
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**ECR Configuration**")
                    ecr_K = gr.Slider(3, 7, value=5, step=1, label="K (Number of candidates)")
                    ecr_H = gr.Slider(2, 4, value=3, step=1, label="H (Horizon steps)")
                    ecr_tau = gr.Slider(0.0, 1.0, value=0.65, step=0.05, label="τ_CCI (Coherence threshold)")
                
                with gr.Column():
                    gr.Markdown("**Control Probe Configuration**")
                    cp_tau = gr.Slider(0.0, 1.0, value=0.40, step=0.05, label="τ (Admissibility threshold)")
                    cp_Theta = gr.Slider(0.5, 3.0, value=2.0, step=0.1, label="Θ (Cumulative risk threshold)")
                
                with gr.Column():
                    gr.Markdown("**IFCS Configuration**")
                    ifcs_rho = gr.Slider(0.0, 1.0, value=0.40, step=0.05, label="ρ (Commitment threshold)")
                    ifcs_lambda_e = gr.Slider(0.0, 1.0, value=0.40, step=0.05, label="λ_e (Evidential weight)")
                    ifcs_lambda_s = gr.Slider(0.0, 1.0, value=0.30, step=0.05, label="λ_s (Scope weight)")
                    ifcs_lambda_a = gr.Slider(0.0, 1.0, value=0.30, step=0.05, label="λ_a (Authority weight)")
            
            init_button = gr.Button("🔧 Initialize System", variant="primary")
            init_status = gr.Textbox(label="Status", interactive=False)
            
            init_button.click(
                fn=web_app.initialize_system,
                inputs=[
                    api_key_input,
                    ecr_K, ecr_H, ecr_tau,
                    cp_tau, cp_Theta,
                    ifcs_rho, ifcs_lambda_e, ifcs_lambda_s, ifcs_lambda_a
                ],
                outputs=init_status
            )
            
            gr.Markdown("---")
            gr.Markdown("### Process a Query")
            
            with gr.Row():
                with gr.Column():
                    prompt_input = gr.Textbox(
                        label="Prompt",
                        placeholder="Enter your query here...",
                        lines=4
                    )
                    context_input = gr.Textbox(
                        label="Context (optional)",
                        placeholder="Additional context or grounding information...",
                        lines=2
                    )
            
            process_button = gr.Button("▶️ Process Query", variant="primary")
            
            with gr.Row():
                with gr.Column():
                    baseline_output = gr.Textbox(
                        label="🔴 Baseline Output (Unregulated)",
                        lines=10,
                        interactive=False
                    )
                
                with gr.Column():
                    regulated_output = gr.Textbox(
                        label="🟢 Regulated Output (Trilogy)",
                        lines=10,
                        interactive=False
                    )
            
            with gr.Row():
                baseline_time = gr.Textbox(
                    label="⏱️ Baseline Time",
                    interactive=False
                )
                regulated_time = gr.Textbox(
                    label="⏱️ Regulated Time",
                    interactive=False
                )
                delta_time = gr.Textbox(
                    label="Δ Time (Regulated - Baseline)",
                    interactive=False
                )
            
            comparison_output = gr.Markdown(label="Comparison Analysis")
            
            process_button.click(
                fn=web_app.process_query,
                inputs=[prompt_input, context_input],
                outputs=[baseline_output, regulated_output, comparison_output, baseline_time, regulated_time, delta_time, gr.State()]
            )
        
        with gr.Tab("📋 Test Cases"):
            gr.Markdown("### Load and test cases from the 36-drift taxonomy")
            
            # Create test case dropdown
            test_case_options = [f"{tc['id']}: {tc['category']}" for tc in TEST_CASES_36_TAXONOMY]
            
            test_case_dropdown = gr.Dropdown(
                choices=test_case_options,
                label="Select Test Case",
                value=test_case_options[0] if test_case_options else None
            )
            
            load_test_button = gr.Button("📥 Load Test Case")
            
            test_info = gr.Markdown()
            
            test_prompt = gr.Textbox(
                label="Test Prompt",
                lines=4,
                interactive=False
            )
            
            run_test_button = gr.Button("▶️ Run Test", variant="primary")
            
            with gr.Row():
                with gr.Column():
                    test_baseline = gr.Textbox(
                        label="Baseline Output",
                        lines=10,
                        interactive=False
                    )
                
                with gr.Column():
                    test_regulated = gr.Textbox(
                        label="Regulated Output",
                        lines=10,
                        interactive=False
                    )
            
            with gr.Row():
                test_baseline_time = gr.Textbox(
                    label="⏱️ Baseline Time",
                    interactive=False
                )
                test_regulated_time = gr.Textbox(
                    label="⏱️ Regulated Time",
                    interactive=False
                )
                test_delta_time = gr.Textbox(
                    label="Δ Time (Regulated - Baseline)",
                    interactive=False
                )
            
            test_comparison = gr.Markdown()
            
            def load_test(selection):
                if selection:
                    test_id = selection.split(':')[0]
                    prompt, info = web_app.load_test_case(test_id)
                    return prompt, info, "", "", "", "", "", ""
                return "", "", "", "", "", "", "", ""
            
            load_test_button.click(
                fn=load_test,
                inputs=test_case_dropdown,
                outputs=[test_prompt, test_info, test_baseline, test_regulated, test_comparison, test_baseline_time, test_regulated_time, test_delta_time]
            )
            
            def run_test(selection):
                if selection:
                    test_id = selection.split(':')[0]
                    return web_app.process_test_case(test_id)
                return ("Please select a test case.", "", "", "error")
            
            run_test_button.click(
                fn=run_test,
                inputs=test_case_dropdown,
                outputs=[test_baseline, test_regulated, test_comparison, test_baseline_time, test_regulated_time, test_delta_time, gr.State()]
            )
        
        with gr.Tab("📤 Batch Processing"):
            gr.Markdown("### Upload a file with prompts (one per line) for batch processing")
            
            file_upload = gr.File(
                label="Upload Prompts File (.txt)",
                file_types=[".txt"]
            )
            
            batch_button = gr.Button("🔄 Process Batch", variant="primary")
            
            batch_status = gr.Textbox(
                label="Batch Processing Status",
                lines=5,
                interactive=False
            )
            
            batch_results = gr.File(label="Download Results")
            
            def process_batch(file):
                if file is None:
                    return "Please upload a file", None
                
                try:
                    with open(file.name, 'r', encoding='utf-8') as f:
                        prompts = [line.strip() for line in f if line.strip()]
                    
                    results = []
                    status_lines = []
                    
                    for i, prompt in enumerate(prompts, 1):
                        status_lines.append(f"[{i}/{len(prompts)}] Processing: {prompt[:50]}...")
                        
                        baseline, regulated, comparison = web_app.app.process_single(prompt)
                        
                        results.append({
                            'prompt': prompt,
                            'baseline': baseline,
                            'regulated': regulated,
                            'comparison': comparison
                        })
                        
                        time.sleep(1)  # Rate limiting
                    
                    # Save results
                    output_path = 'batch_results.json'
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(results, f, indent=2, default=str)
                    
                    status_lines.append(f"\n✅ Completed {len(results)} queries")
                    
                    return "\n".join(status_lines), output_path
                
                except Exception as e:
                    return f"❌ Batch processing failed: {str(e)}", None
            
            batch_button.click(
                fn=process_batch,
                inputs=file_upload,
                outputs=[batch_status, batch_results]
            )
        
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ## About the Trilogy Framework
            
            This system implements the three-paper trilogy by Arijit Chatterjee on inference-time governance for LLMs:
            
            ### 1. Evaluative Coherence Regulation (ECR)
            - **Purpose**: Select the most coherent response among K candidates
            - **Metrics**: EVB, CR, TS, ES, PD → Composite Coherence Index (CCI)
            - **Action**: Selects best candidate based on internal stability
            
            ### 2. Control Probe (Type-1 and Type-2)
            - **Type-1**: Inference-local admissibility gating
              - Blocks outputs when σ(z) < τ (insufficient evaluative support)
            - **Type-2**: Interaction-level monitoring
              - Detects semantic drift, sycophancy, and cumulative risk
              - Triggers HALT or RESET when R_cum ≥ Θ
            
            ### 3. Inference-Time Commitment Shaping (IFCS)
            - **Purpose**: Regulate commitment strength (certainty, scope, authority)
            - **Components**: ê (evidential), ŝ (scope), â (authority), t̂ (temporal)
            - **Action**: Applies transformation rules Γ when R(z) > ρ
            
            ### Pipeline Order (Non-bypassable)
            ```
            ECR → Control Probe Type-1 → IFCS → [output] → Control Probe Type-2
            ```
            
            ### Key Features
            - ✅ Domain-aware calibration (medical, legal, financial)
            - ✅ Boundary compliance (no mechanism overreach)
            - ✅ Non-generative transformations (preserves semantic content)
            - ✅ Auditable traces for all decisions
            
            ### References
            - Chatterjee, A. (2026a). Control Probe: Inference-time commitment control
            - Chatterjee, A. (2026b). Evaluative Coherence Regulation (ECR)
            - Chatterjee, A. (2026c). Inference-Time Commitment Shaping (IFCS)
            
            ---
            
            **Author**: Arijit Chatterjee (ORCID: 0009-0006-5658-4449)  
            **Implementation**: Claude (Anthropic)  
            **License**: Research use
            """)
    
    return interface


def main():
    """Launch Gradio interface"""
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",  # For Replit deployment
        server_port=7860,
        share=False
    )


if __name__ == '__main__':
    main()
