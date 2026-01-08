/*
	- Identify less important feature.
	- Get predictions with biggest difference (positive & negative).
	- Get one more feature (attempt with three, and prune).
	- Continue until improvement stops (test with validation). 
*/

import { iFeature, iFeaturedVideo, iTrainingVideo } from "./utils/types"
import { addFeature, getFeatures } from "./1.Describe"
import MLR from 'ml-regression-multivariate-linear'
import { reportLogErrors } from "./utils/utils"
import { OLLAMA_MODELS } from "./utils/ollama"
import { mkdirSync, writeFileSync } from "fs"
import { rankVideos } from "./2.Rank"


// Step 1: Initialize features
export const featureVideos = async (training:iTrainingVideo[], largeModel?:OLLAMA_MODELS, smallModel?:OLLAMA_MODELS) => {
    const features = await getFeatures(training, largeModel)
    const rankings = await rankVideos(training, features, smallModel)
    const videos:iFeaturedVideo[] = rankings.map(v => ({
        ...training.find(vid => vid.title === v.title)!,
        features: v.features
    }))

    return { videos, features }
}

// Step 2: Train linear regression model
export const trainFeatureModel = (features:iFeature[], videos:iFeaturedVideo[]) => {
    const FEATURE_KEYS = features.map(f => f.name)
    const X = videos.map(v => [
        ...v.recentViews,
        ...FEATURE_KEYS.map(key => v.features[key] || 0)
    ])
    
    const Y = videos.map(v => [Math.log(v.views + 1)])
    const mlr = new MLR(X, Y)

    return {mlr, X, Y}
}

// Step 3: Identify least important feature
export const updateFeatures = (features:iFeature[], mlr:MLR) => {
    const FEATURE_KEYS = features.map(f => f.name)
    const [_, ...featureCoefficients] = mlr.weights.slice(4, mlr.weights.length - 1)
    if(featureCoefficients.length !== FEATURE_KEYS.length) throw Error("Feature length mismatch")

    console.log('Full model weights', mlr.weights.map(w => w[0].toFixed(4)).join(', '));
    console.log('Feature coefficients', featureCoefficients.map(w => w[0].toFixed(4)).join(', '));

    const featuresValue = FEATURE_KEYS.map((key, i) => ({
        name: key,
        importance: Math.abs(featureCoefficients[i][0])
    }))

    const leastFeatureIdx = featuresValue.reduce((minIdx, curr, idx, arr) => 
        curr.importance < arr[minIdx].importance ? idx : minIdx
    , 0)

    const bestFeatures = features.filter((f, i) => i !== leastFeatureIdx)
    const worstFeature = features[leastFeatureIdx]

    console.log(`Removing least important feature: ${worstFeature.name} (importance: ${featuresValue[leastFeatureIdx].importance.toFixed(4)})`)
    return { bestFeatures, worstFeature }
}

export const getWorstPredictions = (videos:iFeaturedVideo[], mlr:MLR, X:number[][], Y:number[][]) => {
    const predictions = mlr.predict(X).map(([pred]) => pred)
    const diffs = predictions.map((pred, i) => Math.abs(pred - Y[i][0]))

    const sortedDiffs = diffs.slice().sort((a, b) => b - a)
    const worstDiffs = sortedDiffs.slice(0, 20)
    const worstIndices = worstDiffs.map(d => diffs.indexOf(d))

    return worstIndices.map(idx => ({
        title: videos[idx].title,
        actual: Math.exp(Y[idx][0]) - 1,
        predicted: Math.exp(predictions[idx]) - 1,
        diff: Math.abs(Math.exp(predictions[idx]) - Math.exp(Y[idx][0])),
        delta: predictions[idx] - Y[idx][0]
    }))
}


export const trainReflexion = async (training:iTrainingVideo[], validation:iTrainingVideo[]) => {
    const { videos, features } = await featureVideos(training)
    const {mlr, X, Y} = trainFeatureModel(features, videos)

    const {bestFeatures} = updateFeatures(features, mlr)
    const worstPredictions = getWorstPredictions(videos, mlr, X, Y)

    const newFeature = await addFeature(bestFeatures, worstPredictions, [])
    const newRankings = await rankVideos(training, [newFeature])

    const newVideos:iFeaturedVideo[] = videos.map(v => ({ ...v,
        features: { 
            ...v.features,
            [newFeature.name]: newRankings.find(r => r.title === v.title)?.features[newFeature.name] || 0
        }
    }))
    const updatedFeatures = [...bestFeatures, newFeature]

    const newModel = trainFeatureModel(updatedFeatures, newVideos)

    const allFeatures = [...features, newFeature]
    const validation_rankings = await rankVideos(validation, allFeatures)

    const validation_Y = validation.map(v => [Math.log(v.views + 1)])

    const validation_X1 = validation_rankings.map(v => {
        const video = validation.find(vid => vid.title === v.title)!
        return ([
            video.recentViews.reduce((a, b) => a + b, 0) / video.recentViews.length,
            ...features.map(f => v.features[f.name] || 0)
        ])
    })      
    const validation_predictions1 = mlr.predict(validation_X1).map(([pred]) => pred)
    const trainingResults = await reportLogErrors(validation_predictions1, validation_Y.map(y => y[0]), mlr)

    const validation_X2 = validation_rankings.map(v => {
        const video = validation.find(vid => vid.title === v.title)!
        return ([
            video.recentViews.reduce((a, b) => a + b, 0) / video.recentViews.length,
            ...updatedFeatures.map(f => v.features[f.name] || 0)
        ])
    })
    const validation_predictions2 = newModel.mlr.predict(validation_X2).map(([pred]) => pred)
    const validationResults = await reportLogErrors(validation_predictions2, validation_Y.map(y => y[0]))


    return { trainingResults, validationResults, features: updatedFeatures }

}



/*
Compares two models: one with current features, one with the new feature.
Uses validation set to determine when to stop adding features.
Receives one model, a set of features, and training & validation data.
Trains a new model with an additional feature, and compares performance.
*/

interface iReflexionStepInput {
    features: iFeature[],
    videos: iFeaturedVideo[],
    failedFeatures: iFeature[],
    dir: string,
    index: number
}
export const reflexionStep = async ({features, videos, failedFeatures, dir, index}:iReflexionStepInput) => {
    const shuffledData = [...videos].slice().sort(() => Math.random() - 0.5);
    const trainingSize = Math.floor(shuffledData.length * 0.8);
    const training = shuffledData.slice(0, trainingSize);
    const validation = shuffledData.slice(trainingSize);

    console.log(`Reflexion step #${index} with ${features.length} features: ${features.map(f => f.name).join(', ')}`);
    const {mlr, X, Y} = trainFeatureModel(features, training)
    const {bestFeatures, worstFeature} = updateFeatures(features, mlr)

    // Get new feature.
    const worstPredictions = getWorstPredictions(training, mlr, X, Y)
    const hasWorstFeature = failedFeatures.some(f => f.name === worstFeature.name)
    const fullFailedFeatures = hasWorstFeature ? failedFeatures : [...failedFeatures, worstFeature]
    const newFeature = await addFeature(bestFeatures, worstPredictions, fullFailedFeatures)
    const newRankings = await rankVideos(training, [newFeature])

    console.log(`Training new model with feature: ${newFeature.name}`);
    const updatedFeatures = [...bestFeatures, newFeature]
    const newVideos:iFeaturedVideo[] = training.map(v => ({ ...v,
        features: { 
            ...v.features,
            [newFeature.name]: newRankings.find(r => r.title === v.title)?.features[newFeature.name] || 0
        }
    }))

    // Train new model.
    const {mlr:newModel} = trainFeatureModel(updatedFeatures, newVideos)

    // Validate old model.
    const validation_Y = validation.map(v => [Math.log(v.views + 1)])
    const previousModel_X = validation.map(v => ([
        ...v.recentViews,
        ...features.map(f => v.features[f.name] || 0)
    ]))

    console.log(`Evaluating previous model on validation set...`);
    const previousModel_predictions = mlr.predict(previousModel_X).map(([pred]) => pred)
    const previousModelResults = await reportLogErrors(previousModel_predictions, validation_Y.map(y => y[0]), mlr)

    const validation_rankings = await rankVideos(validation, [newFeature])
    const validationVideos:iFeaturedVideo[] = validation.map(v => ({ ...v,
        features: { ...v.features,
            [newFeature.name]: validation_rankings.find(r => r.title === v.title)?.features[newFeature.name] || 0
        }
    }))

    const newModel_X = validationVideos.map(v => ([
        ...v.recentViews,
        ...updatedFeatures.map(f => v.features[f.name] || 0)
    ]))

    console.log(`Evaluating new model on validation set...`);
    const newModel_predictions = newModel.predict(newModel_X).map(([pred]) => pred)
    const newModelResults = await reportLogErrors(newModel_predictions, validation_Y.map(y => y[0]), newModel)

    const failedFeature:iFeature = features.find(f => !updatedFeatures.some(uf => uf.name === f.name))!

    writeFileSync(`${dir}/features-${index}.json`, JSON.stringify({
        features: updatedFeatures,
        originalFeatures: features,
        failedFeatures,
        newFeature,
        failedFeature
    }, null, 2))

    writeFileSync(`${dir}/videos-${index}.json`, JSON.stringify({
        rankings: newVideos,
        validationRankings: validationVideos,
        trainingVideos: training,
        validationVideos: validation
    }, null, 2))

    writeFileSync(`${dir}/results-${index}.json`, JSON.stringify({
        previousModelResults,
        newModelResults
    }, null, 2))

    return { 
        previousModelResults, 
        newModelResults, 
        features: updatedFeatures, 
        videos: [newVideos, validationVideos].flat(), 
        failedFeature,
        newFeature
    }
}


const DIR = './data/reflexion'
export const runReflexion = async (videos:iTrainingVideo[], dir:string) => {
    mkdirSync(`${dir}/reflexion`, { recursive: true })

    let { features, videos: featuredVideos } = await featureVideos(videos)
    console.log(`Initial features: ${features.map(f => f.name).join(', ')}`)
    console.log(`Starting reflexion process with ${videos.length} training videos.`)

    const failedFeatures:iFeature[] = []
    for (let index = 0; index < 10; index++) {
        try {
            const reflexionInput = { features, videos: featuredVideos, failedFeatures, dir, index }
            const results = await reflexionStep(reflexionInput)
            featuredVideos = results.videos

            const { newModelResults, previousModelResults } = results
            const improved = newModelResults.averageLogError < previousModelResults.averageLogError

            if (improved){
                features = results.features
                failedFeatures.push(results.failedFeature)

                console.log(`Improved. New features: ${results.features.map(f => f.name).join(', ')}`);
            } else {
                failedFeatures.push(results.newFeature)
                console.log(`No improvement. Failed features: ${failedFeatures.map(f => f.name).join(', ')}`);
            }

            console.log('')
        } catch (err) {
            console.error(`Error during reflexion step ${index}:`, err);
        }
    }

    writeFileSync(`${dir}/features.json`, JSON.stringify({features, failedFeatures}, null, 2))
    writeFileSync(`${dir}/reflexionVideos.json`, JSON.stringify(featuredVideos, null, 2))

    console.log(`Reflexion process completed. Final features: ${features.map(f => f.name).join('\n')}`)
    console.log(`Failed features: ${failedFeatures.map(f => f.name).join('\n')}`)
    return { features, videos: featuredVideos }
}
