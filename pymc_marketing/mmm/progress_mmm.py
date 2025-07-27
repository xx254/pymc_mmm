"""Progress tracking MMM model."""

import sys
from typing import Any, Dict, Optional, Union
import numpy as np
from pymc_marketing.mmm import MMM
from pymc_marketing.mmm import GeometricAdstock, MichaelisMentenSaturation


class ProgressMMM(MMM):
    """MMM model with progress tracking during training."""
    
    def __init__(self, 
                 date_column: str = "date",
                 channel_columns: Optional[list] = None,
                 control_columns: Optional[list] = None,
                 adstock: Optional[Any] = None,
                 saturation: Optional[Any] = None,
                 time_varying_intercept: bool = False,
                 time_varying_media: bool = False,
                 model_config: dict = None,
                 sampler_config: dict = None,
                 validate_data: bool = True,
                 yearly_seasonality: Optional[int] = None,
                 adstock_first: bool = True,
                 dag: Optional[str] = None,
                 treatment_nodes: Optional[list] = None,
                 outcome_node: Optional[str] = None,
                 **kwargs):
        """Initialize the model with progress tracking.
        
        Args:
            All the standard MMM parameters plus any additional kwargs
        """
        # Set default transformations if none provided
        if adstock is None:
            adstock = GeometricAdstock(l_max=24)
        if saturation is None:
            saturation = MichaelisMentenSaturation()
            
        # Pass all arguments to the parent MMM class
        super().__init__(
            date_column=date_column,
            channel_columns=channel_columns,
            control_columns=control_columns,
            adstock=adstock,
            saturation=saturation,
            time_varying_intercept=time_varying_intercept,
            time_varying_media=time_varying_media,
            model_config=model_config,
            sampler_config=sampler_config,
            validate_data=validate_data,
            yearly_seasonality=yearly_seasonality,
            adstock_first=adstock_first,
            dag=dag,
            treatment_nodes=treatment_nodes,
            outcome_node=outcome_node,
            **kwargs
        )

    def fit(self, X, y, **kwargs):
        """Fit the model with progress tracking.
        
        Args:
            X: Feature matrix
            y: Target variable
            **kwargs: Additional arguments passed to pymc.sample
        """
        # Set up progress bar callback
        class ProgressCallback:
            def __init__(self, total_draws):
                self.total = total_draws * kwargs.get("chains", 4)  # Total steps
                self.current = 0
                self.last_update = 0
            
            def __call__(self, *args, **kwargs):
                """Update progress."""
                self.current += 1
                progress = int(100 * self.current / self.total)
                
                if progress > self.last_update:
                    self.last_update = progress
                    draws = kwargs.get("draws", 1000)
                    chain = kwargs.get("chain", 0)
                    draw = kwargs.get("draw", 0)
                    
                    msg = f"\r🔄 训练进度: {progress}% "
                    msg += f"[链 {chain + 1}/{kwargs.get('chains', 4)}, "
                    msg += f"采样 {draw + 1}/{draws}]"
                    
                    sys.stdout.write(msg)
                    sys.stdout.flush()

        # Calculate total steps
        total_draws = kwargs.get("draws", 1000)
        progress_callback = ProgressCallback(total_draws)

        # Add callback to sampling arguments
        if "callback" in kwargs:
            old_callback = kwargs["callback"]
            def combined_callback(*args, **cb_kwargs):
                progress_callback(*args, **cb_kwargs)
                old_callback(*args, **cb_kwargs)
            kwargs["callback"] = combined_callback
        else:
            kwargs["callback"] = progress_callback

        print("\n🚀 开始模型训练...")
        print(f"总采样次数: {total_draws}")
        print(f"链数: {kwargs.get('chains', 4)}")
        print(f"预热期: {kwargs.get('tune', 1000)} 步")
        
        # Call parent's fit method
        result = super().fit(X, y, **kwargs)
        
        print("\n✅ 模型训练完成!")
        
        # Check divergences
        if hasattr(self, 'idata'):
            divergences = self.idata["sample_stats"]["diverging"].sum().item()
            print(f"🔍 发散链数: {divergences}")
            
            # Additional convergence diagnostics
            try:
                import arviz as az
                summary = az.summary(self.idata)
                print("\n📊 模型收敛诊断:")
                print(f"Effective Sample Size (ESS): {summary['ess_bulk'].mean():.1f}")
                print(f"R-hat 统计量: {summary['r_hat'].mean():.3f}")
            except Exception as e:
                print(f"无法计算收敛诊断: {e}")
        
        return result
