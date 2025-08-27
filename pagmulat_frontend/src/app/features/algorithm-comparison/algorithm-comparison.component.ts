import { Component, OnInit } from '@angular/core';
import { GeneratedRulesComparisonService, Rule  } from 'src/app/services/generated-rules-comparison.service';

@Component({
  selector: 'app-algorithm-comparison',
  templateUrl: './algorithm-comparison.component.html',
  styleUrls: ['./algorithm-comparison.component.css'],
})
export class AlgorithmComparisonComponent implements OnInit {
  minSupport = 0;
  minConfidence = 0;
  minLift = 0;
  aprioriRules: Rule[] = [];
  fpGrowthRules: Rule[] = [];

  constructor(private generatedRulesComparisonService: GeneratedRulesComparisonService) {}

  ngOnInit(): void {
    this.generatedRulesComparisonService.getRules().subscribe(data => {
      console.log(data)
      this.minSupport = data.minSupport;
      this.minConfidence = data.minConfidence;
      this.minLift = data.minLift;
      this.aprioriRules = data.aprioriRules;
      this.fpGrowthRules = data.fpGrowthRules;
    });
  }
}
